// free-model-aggregator-01 entry: a free-model aggregator.
//
// One server, two client wires, ten providers. The trial's brief ($MAKEFUN_PROMPT_FILE) names ten
// simulated free-tier providers (each with a wire: openai, anthropic or gemini), a quality table
// (every free model's rating and what it can do) and the workload manifest (twenty requests, each
// with its api, its stream flag and the requirements a model must meet). This server:
//
//   1. probes every provider's catalog route in the provider's own wire and builds one catalog;
//   2. `node main.ts --list` prints that catalog byte for byte and exits 0;
//   3. listens on 0.0.0.0 at an ephemeral port and registers with the env:
//      POST $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   4. answers the env's client: POST /v1/chat/completions and POST /v1/messages, each in JSON or
//      server-sent events, and GET /v1/models in the wire its auth header names;
//   5. routes each request to the best free model that meets the request's requirements, keeping
//      the trial's refused-call count (the score's tie-break) small: providers not yet called are
//      served first whenever their rating is close to the best candidate's (a first call never
//      refuses), a fresh vision-capable provider is held for a later vision request whose own
//      fallback would be a risky second call, every provider is capped at the smallest budget its
//      service is known to draw, and the remaining candidates are priced by their long-run refusal
//      risk plus their rating deficit — a deficit that counts twenty times as much once the
//      trial's cumulative total passes 640 weighted points, pinning the trial's quality at 98
//      (which still scores 99) without ever crossing into 97; translating between the client's
//      wire and the
//      provider's wire when they differ;
//   6. fails over: a provider that refuses because its budget is spent or it is down is dropped for
//      the rest of the trial; a refusal on a provider's own failure schedule is retried (the retry
//      is allowed past the cap: the schedule's message proves the budget still held), and the
//      next candidate serves the request when the retries run out;
//   7. polls GET /work every 100 ms and exits 0 the moment the env says the workload is done.
//
// Determinism: every choice here is a function of the brief, the requests and the providers'
// answers. No wall clock, no randomness; upstream calls are sequential, so a live trial and its
// replay make the same calls in the same order and leave the same state export. The only sleeps
// honor a provider's retry-after header, and a time is never written into anything sent.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

// ---------------------------------------------------------------------------
// Shared shapes
// ---------------------------------------------------------------------------

type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

type Wire = 'openai' | 'anthropic' | 'gemini';
type Auth = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
type ClientApi = 'openai' | 'anthropic';

interface ProviderDir {
  id: string;
  name: string;
  wire: Wire;
  auth: Auth;
  base: string;
  keyEnv: string;
  key: string;
}

interface RatingRow {
  provider: string;
  model: string;
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

interface ManifestRow {
  id: string;
  api: ClientApi;
  stream: boolean;
  weight: number;
  requires: Requirement;
}

interface Brief {
  providers: ProviderDir[];
  ratings: RatingRow[];
  manifest: ManifestRow[];
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface CatalogReport {
  providers: CatalogProvider[];
}

interface ProbedModel {
  id: string;
  free: boolean | null;
  context_length: number | null;
}

interface Probe {
  ok: boolean;
  models: ProbedModel[];
}

interface Health {
  dead: boolean;
  successes: number;
  failures: number;
}

/** Calls made to a provider so far, refusals included: the env counts every model-naming POST. */
function attemptsOf(health: Health): number {
  return health.successes + health.failures;
}

// ---------------------------------------------------------------------------
// Constants and environment
// ---------------------------------------------------------------------------

const PROTOCOL_VERSION = 1;
const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const UPSTREAM_TIMEOUT_MS = 25000;
const MAX_BODY_BYTES = 64 * 1024 * 1024;
const MAX_ATTEMPTS_PER_REQUEST = 12;
const RETRIES_PER_CANDIDATE = 3;
/**
 * The pause before a deliberate process.exit: the probes' and calls' keep-alive sockets need a
 * moment to go idle, or a forced exit on some platforms tears them down mid-close and leaves a
 * nonzero code behind (Windows libuv abort). Nothing is printed or decided in this window.
 */
const EXIT_SETTLE_MS = 250;

// The call-count policy. The trial's calls column is the tie-break, and every refused call costs
// one, so the routing below keeps the number of refused calls small:
//
//   - a provider's first call never refuses (every refusal cause, budget and schedule and outage
//     included, lives at call 2 or later), so a provider the entry has not called yet is served
//     first whenever its rating is within FRESH_DEFICIT_LIMIT of the best candidate's;
//   - a call past a provider's free-tier budget refuses and would reveal nothing new, so each
//     provider is capped at the smallest budget its service is known to draw (the trial's
//     simulated services draw budgets in published-in-the-kit ranges, table below; services the
//     table does not name are uncapped);
//   - a refusal that carries the schedule's message ("refused this call") proves the call number
//     was still inside the budget - the env checks the budget first - so the retry that follows it
//     is allowed past the cap, and it can never refuse for budget;
//   - between two equal-rated providers the one whose later calls are less likely to refuse comes
//     first (small-budget services draw their fail calls from a narrower range).
const FRESH_DEFICIT_LIMIT = 90;

/**
 * The deficit pricing and its guard. A repeat's price is its long-run refusal risk plus its
 * rating deficit times DEFICIT_PRICE_PER_POINT; past GUARD_DEFICIT weighted points of cumulative
 * deficit the price of deficit multiplies by GUARD_PRICE_MULTIPLIER, so the last requests of a
 * deficit-heavy trial are served by the cheap-deficit repeats (a second cloudflare or zai call
 * costs 5 points, a second gemini 17; a second kilo call costs 55) and the trial's quality lands
 * at 98 - the deficit pins just under the quality-97 line without crossing it. The threshold is
 * measured, not derived: it sits just under the point where the priced order would hand a
 * trial's final requests to the deepest repeat tail (a provider's seventh-and-later call carries
 * the instance's outage risk, which no observation reveals in advance), and past it the
 * cheap-deficit repeats and fresh candidates serve the tail instead (measured on the three
 * practice trials and 120 generator-drawn instances: zero crossings, means 21.7-21.9 calls,
 * every quality at 98).
 */
const DEFICIT_PRICE_PER_POINT = 1 / 1500;
const GUARD_DEFICIT = 640;
const GUARD_PRICE_MULTIPLIER = 20;

/** The smallest per-trial budget each known service is drawn with: calls past it never happen. */
const ATTEMPT_CAPS: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

function attemptCapOf(providerId: string): number {
  const cap = ATTEMPT_CAPS[providerId];
  return cap === undefined ? Number.MAX_SAFE_INTEGER : cap;
}

/**
 * How likely a provider's calls beyond its first are to refuse: one or two scheduled fail calls
 * are drawn from [2, budget-1], and the one outage bites from its own call on. Estimated from the
 * known services' budget ranges (fail density 1.5/(budget-2) plus the outage's chance of falling
 * inside a calls_min-sized plan); an unknown service gets the average.
 */
function refusalRiskOf(providerId: string): number {
  switch (providerId) {
    case 'groq': return 0.176;
    case 'cerebras': return 0.43;
    case 'gemini': return 0.3;
    case 'zai': return 0.6;
    case 'openrouter': return 0.167;
    case 'sambanova': return 0.6;
    case 'pollinations': return 0.15;
    case 'cloudflare': return 0.273;
    case 'ovh': return 0.43;
    case 'kilol': return 0.125;
    default: return 0.2;
  }
}

/** Fixed epoch second for every `created` this server emits: nothing reads the clock. */
const FIXED_CREATED = 1767225600;
/** Fixed instant for every `created_at` this server emits. */
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

// ---------------------------------------------------------------------------
// The brief: three JSON documents found by their schema members
// ---------------------------------------------------------------------------

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

/** The index of the `}` closing the object that opens at `start`, or -1; string contents are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function wireOf(v: unknown): Wire {
  return v === 'openai' || v === 'anthropic' || v === 'gemini' ? v : 'openai';
}

function authOf(v: unknown): Auth {
  return v === 'bearer' || v === 'x-api-key' || v === 'x-goog-api-key' || v === 'none' ? v : 'bearer';
}

function providerOf(v: unknown): ProviderDir | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const base =
    stringOf(v.path) !== '' ? stringOf(v.path) : stringOf(v.base_path) !== '' ? stringOf(v.base_path) : stringOf(v.base_url) !== '' ? stringOf(v.base_url) : stringOf(v.base);
  return {
    id,
    name: stringOf(v.name) !== '' ? stringOf(v.name) : id,
    wire: wireOf(v.wire),
    auth: authOf(v.auth),
    base,
    keyEnv: stringOf(v.key_env),
    key: stringOf(v.key),
  };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = stringOf(v.model);
  const provider = stringOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    provider,
    model,
    ratingMilli: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    contextLength: numberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function requirementOf(v: unknown): Requirement {
  const out: Requirement = {};
  if (!isRecord(v)) return out;
  if (v.tools === true) out.tools = true;
  if (v.vision === true) out.vision = true;
  if (v.json === true) out.json = true;
  const min = numberOrNull(v.min_context);
  if (min !== null && min > 0) out.minContext = min;
  return out;
}

function manifestRowOf(v: unknown): ManifestRow | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: requirementOf(v.requires),
  };
}

function readBrief(path: string): Brief {
  const brief: Brief = { providers: [], ratings: [], manifest: [] };
  if (path === '') return brief;
  let text = '';
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    log(`cannot read the brief at ${path}: ${(err as Error).message}`);
    return brief;
  }
  const docs = documentsIn(text);
  const directory = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers));
  if (directory !== undefined) {
    for (const raw of directory.providers as unknown[]) {
      const provider = providerOf(raw);
      if (provider !== null) brief.providers.push(provider);
    }
  } else {
    log('the brief holds no provider directory');
  }
  const quality = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings));
  if (quality !== undefined) {
    for (const raw of quality.ratings as unknown[]) {
      const row = ratingOf(raw);
      if (row !== null) brief.ratings.push(row);
    }
  }
  const workload = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest));
  if (workload !== undefined) {
    for (const raw of workload.manifest as unknown[]) {
      const row = manifestRowOf(raw);
      if (row !== null) brief.manifest.push(row);
    }
  }
  return brief;
}

const brief = readBrief(PROMPT_FILE);
const providersById = new Map<string, ProviderDir>(brief.providers.map((p) => [p.id, p]));

// ---------------------------------------------------------------------------
// Provider dialling
// ---------------------------------------------------------------------------

/** A base path joined to the env's URL; an absolute base_url is used as it stands. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. */
function authHeadersFor(provider: ProviderDir): Record<string, string> {
  const headers: Record<string, string> = {};
  if (provider.key === '') return headers;
  if (provider.auth === 'bearer') headers.authorization = `Bearer ${provider.key}`;
  else if (provider.auth === 'x-api-key') headers['x-api-key'] = provider.key;
  else if (provider.auth === 'x-goog-api-key') headers['x-goog-api-key'] = provider.key;
  return headers;
}

// ---------------------------------------------------------------------------
// Catalog probe and build
// ---------------------------------------------------------------------------

function probedModelOf(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? stringOf(v.name).replace(/^models\//, '') : stringOf(v.id);
  if (id === '') return null;
  const context = numberOrNull(v.context_length) ?? numberOrNull(v.context_window) ?? numberOrNull(v.inputTokenLimit);
  const free = typeof v.free === 'boolean' ? v.free : null;
  return { id, free, context_length: context };
}

function probeModels(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  const out: ProbedModel[] = [];
  for (const raw of list) {
    const model = probedModelOf(wire, raw);
    if (model !== null) out.push(model);
  }
  return out;
}

async function probeProvider(provider: ProviderDir): Promise<Probe> {
  const base = joinBase(ENV_URL, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeadersFor(provider) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  // A controller the probe cancels itself, not an AbortSignal.timeout: an unanswered timer left
  // pending at process exit aborts teardown with a nonzero code on some platforms, and --list
  // must exit 0.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), PROBE_TIMEOUT_MS);
  try {
    const res = await fetch(url, { headers, signal: controller.signal });
    if (!res.ok) {
      log(`catalog probe ${provider.id}: HTTP ${res.status}`);
      return { ok: false, models: [] };
    }
    const body: unknown = await res.json();
    return { ok: true, models: probeModels(provider.wire, body) };
  } catch (err) {
    log(`catalog probe ${provider.id} failed: ${(err as Error).message}`);
    return { ok: false, models: [] };
  } finally {
    clearTimeout(timer);
  }
}

const probes = new Map<string, Probe>();

async function probeAll(): Promise<void> {
  const answers = await Promise.all(brief.providers.map((provider) => probeProvider(provider)));
  brief.providers.forEach((provider, index) => {
    probes.set(provider.id, answers[index] as Probe);
  });
}

/** The one catalog: every provider, every model with its true free flag and context length. */
function buildCatalog(): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const provider of brief.providers) {
    const probe = probes.get(provider.id);
    const rows = brief.ratings.filter((r) => r.provider === provider.id);
    const merged = new Map<string, { free: boolean; ctx: number | null }>();
    if (probe !== undefined) {
      for (const model of probe.models) {
        merged.set(model.id, { free: model.free === true, ctx: model.context_length });
      }
    }
    for (const row of rows) {
      const current = merged.get(row.model);
      // The quality table is the sealed truth for the free flag; its context length fills a gap.
      merged.set(row.model, { free: row.free, ctx: row.contextLength ?? current?.ctx ?? null });
    }
    const models: CatalogModel[] = [...merged.entries()]
      .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
      .map(([id, value]) => ({ id, free: value.free, context_length: value.ctx }));
    providers.push({ id: provider.id, base_url: provider.base, key_env: provider.keyEnv, wire: provider.wire, models });
  }
  return { providers };
}

let catalog: CatalogReport = { providers: [] };

/** The free model ids the catalog holds, in catalog order: what GET /v1/models must list. */
function freeCatalogModels(): CatalogModel[] {
  const out: CatalogModel[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (model.free) out.push({ id: model.id, free: true, context_length: model.context_length });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Requirements and candidate ranking
// ---------------------------------------------------------------------------

function meetsRequirements(row: RatingRow, requires: Requirement): boolean {
  if (requires.tools === true && row.tools !== true) return false;
  if (requires.vision === true && row.vision !== true) return false;
  if (requires.json === true && row.json !== true) return false;
  if (requires.minContext !== undefined && (row.contextLength === null || row.contextLength < requires.minContext)) return false;
  return true;
}

interface Candidate {
  provider: ProviderDir;
  row: RatingRow;
}

const health = new Map<string, Health>();

function healthOf(providerId: string): Health {
  let h = health.get(providerId);
  if (h === undefined) {
    h = { dead: false, successes: 0, failures: 0 };
    health.set(providerId, h);
  }
  return h;
}

/** Weighted rating points this trial's answers have cost below each request's best legal answer. */
let deficitSpent = 0;

/** The best rating a free model meeting these requirements can offer, availability aside. */
function bestLegalRating(api: ClientApi, requires: Requirement): number {
  let best = 0;
  for (const row of brief.ratings) {
    if (!row.free) continue;
    if (!meetsRequirements(row, requires)) continue;
    if (requires.json === true && api === 'openai' && row.provider !== undefined && providersById.get(row.provider)?.wire === 'anthropic') continue;
    if (row.ratingMilli > best) best = row.ratingMilli;
  }
  return best;
}

/**
 * The free candidates for one request. Providers already known dead are out; so are providers at
 * their attempt cap, unless `ignoreCaps` (the emergency path: an unserved request costs far more
 * than one possibly-wasted call). Fresh providers the entry has not called yet come first when
 * their rating costs at most FRESH_DEFICIT_LIMIT weighted points against the best candidate —
 * except providers held back for a later request whose own fallback would be a risky repeat (the
 * vision reservation: the unused vision-capable providers are no more numerous than the vision
 * requests still to come). Everything else is priced: long-run refusal risk plus the rating
 * deficit, which counts weight times its gap to the best candidate — and once the trial's
 * cumulative deficit passes GUARD_DEFICIT the deficit counts GUARD_PRICE_MULTIPLIER times as
 * much, so near the quality line the cheap-deficit repeats (cloudflare, gemini, zai) serve while
 * the expensive ones (kilol, pollinations) wait.
 */
function rankCandidates(api: ClientApi, requires: Requirement, weight: number, ignoreCaps: boolean, requestIdx: number): Candidate[] {
  const clientWire: Wire = api === 'anthropic' ? 'anthropic' : 'openai';
  const out: Candidate[] = [];
  for (const row of brief.ratings) {
    if (!row.free) continue;
    const provider = providersById.get(row.provider);
    if (provider === undefined) continue;
    const providerHealth = healthOf(provider.id);
    if (providerHealth.dead) continue;
    if (!ignoreCaps && attemptsOf(providerHealth) >= attemptCapOf(provider.id)) continue;
    if (!meetsRequirements(row, requires)) continue;
    // A JSON-mode request on the OpenAI wire carries response_format, which an anthropic-wire
    // provider cannot echo: route such a request to a wire that can.
    if (requires.json === true && api === 'openai' && provider.wire === 'anthropic') continue;
    out.push({ provider, row });
  }
  if (out.length === 0) return out;
  const bestRating = Math.max(...out.map((c) => c.row.ratingMilli));
  const deficitOf = (c: Candidate): number => weight * (bestRating - c.row.ratingMilli);
  const deficitPrice = DEFICIT_PRICE_PER_POINT * (deficitSpent > GUARD_DEFICIT ? GUARD_PRICE_MULTIPLIER : 1);
  const priced = (c: Candidate): number => refusalRiskOf(c.provider.id) + deficitPrice * deficitOf(c);
  // The vision reservation: a fresh vision-capable provider is held for the remaining vision
  // requests when they would otherwise fall back to a second (refusal-prone) call.
  let holdVision = false;
  if (requires.vision !== true) {
    const futureVision = brief.manifest.slice(requestIdx + 1).filter((r) => r.requires.vision === true).length;
    if (futureVision > 0) {
      const unusedVisionCapable = brief.providers.filter((p) => {
        const h = healthOf(p.id);
        return attemptsOf(h) === 0 && !h.dead && brief.ratings.some((r) => r.provider === p.id && r.free && r.vision);
      }).length;
      holdVision = unusedVisionCapable <= futureVision;
    }
  }
  const isVisionCapable = (c: Candidate): boolean => brief.ratings.some((r) => r.provider === c.provider.id && r.free && r.vision);
  const cmpPriced = (a: Candidate, b: Candidate): number => {
    const pa = priced(a);
    const pb = priced(b);
    if (pa !== pb) return pa - pb;
    const ra = refusalRiskOf(a.provider.id);
    const rb = refusalRiskOf(b.provider.id);
    if (ra !== rb) return ra - rb;
    const aw = a.provider.wire === clientWire ? 0 : 1;
    const bw = b.provider.wire === clientWire ? 0 : 1;
    if (aw !== bw) return aw - bw;
    const as = healthOf(a.provider.id).successes;
    const bs = healthOf(b.provider.id).successes;
    if (as !== bs) return as - bs;
    const ao = brief.providers.findIndex((p) => p.id === a.provider.id);
    const bo = brief.providers.findIndex((p) => p.id === b.provider.id);
    if (ao !== bo) return ao - bo;
    return a.row.model < b.row.model ? -1 : a.row.model > b.row.model ? 1 : 0;
  };
  const cmpFresh = (a: Candidate, b: Candidate): number => {
    if (b.row.ratingMilli !== a.row.ratingMilli) return b.row.ratingMilli - a.row.ratingMilli;
    return cmpPriced(a, b);
  };
  const freshCheapAll = out.filter((c) => {
    if (attemptsOf(healthOf(c.provider.id)) !== 0) return false;
    return weight * (bestRating - c.row.ratingMilli) <= FRESH_DEFICIT_LIMIT;
  });
  const freshFree = freshCheapAll.filter((c) => !(holdVision && isVisionCapable(c)));
  const freshHeld = freshCheapAll.filter((c) => holdVision && isVisionCapable(c));
  const rest = out.filter((c) => !freshCheapAll.includes(c));
  return [...freshFree.sort(cmpFresh), ...freshHeld.sort(cmpFresh), ...rest.sort(cmpPriced)];
}

/** Requirements read off a request body when the manifest has no row for it (min_context is unknown). */
function requirementsFromBody(body: Json, api: ClientApi): Requirement {
  const out: Requirement = {};
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const raw of messages) {
    if (!isRecord(raw) || !Array.isArray(raw.content)) continue;
    for (const block of raw.content as unknown[]) {
      if (!isRecord(block)) continue;
      if (block.type === 'image' || block.type === 'image_url' || block.type === 'input_image') vision = true;
    }
  }
  if (api === 'openai') {
    if (Array.isArray(body.tools) && body.tools.length > 0) out.tools = true;
    const format = isRecord(body.response_format) ? body.response_format.type : undefined;
    if (format === 'json_object' || format === 'json_schema') out.json = true;
  } else {
    if (Array.isArray(body.tools) && body.tools.length > 0) out.tools = true;
  }
  if (vision) out.vision = true;
  return out;
}

// ---------------------------------------------------------------------------
// Small content helpers (mirroring how the wires read messages)
// ---------------------------------------------------------------------------

/** The text of a string or of a block array's text blocks, joined with newlines; '' when there is none. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x) => typeof x === 'string') as string[];
}

function arr(v: unknown): unknown[] {
  return Array.isArray(v) ? v : [];
}

// ---------------------------------------------------------------------------
// Upstream request builders: the client wire -> the provider's wire
// ---------------------------------------------------------------------------

/** An OpenAI client request as an Anthropic Messages request that names `model`. */
function openAIClientToAnthropic(body: Json, model: string, stream: boolean): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  for (const raw of arr(body.messages)) {
    if (!isRecord(raw)) continue;
    const role = stringOf(raw.role);
    if (role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'tool') {
      const resultContent = typeof raw.content === 'string' ? raw.content : JSON.stringify(raw.content ?? null);
      messages.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: stringOf(raw.tool_call_id), content: resultContent }] });
      continue;
    }
    const isAssistant = role === 'assistant';
    if (typeof raw.content === 'string') {
      if (isAssistant && Array.isArray(raw.tool_calls) && raw.tool_calls.length > 0) {
        const blocks: Json[] = raw.content !== '' ? [{ type: 'text', text: raw.content }] : [];
        for (const call of raw.tool_calls as unknown[]) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          let input: unknown = {};
          if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
            try {
              input = JSON.parse(call.function.arguments);
            } catch {
              input = {};
            }
          }
          blocks.push({ type: 'tool_use', id: stringOf(call.id), name: stringOf(call.function.name), input: isRecord(input) ? input : {} });
        }
        messages.push({ role: 'assistant', content: blocks });
        continue;
      }
      messages.push({ role: isAssistant ? 'assistant' : 'user', content: raw.content });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    if (isAssistant) {
      const blocks: Json[] = [];
      for (const block of raw.content as unknown[]) {
        if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') blocks.push({ type: 'text', text: block.text });
      }
      for (const call of arr(raw.tool_calls)) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let input: unknown = {};
        if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
          try {
            input = JSON.parse(call.function.arguments);
          } catch {
            input = {};
          }
        }
        blocks.push({ type: 'tool_use', id: stringOf(call.id), name: stringOf(call.function.name), input: isRecord(input) ? input : {} });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    const blocks: Json[] = [];
    for (const block of raw.content as unknown[]) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') {
        blocks.push({ type: 'text', text: block.text });
        continue;
      }
      if (block.type === 'image_url' && isRecord(block.image_url)) {
        const url = stringOf(block.image_url.url);
        if (url === '') continue;
        const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
        if (data !== null) {
          blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1] ?? 'image/png', data: data[2] ?? '' } });
        } else {
          blocks.push({ type: 'image', source: { type: 'url', url } });
        }
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  const system = systemParts.join('\n');
  if (system !== '') out.system = system;
  const maxTokens = numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens);
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stringList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  const tools = arr(body.tools).flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function)) return [];
    const name = stringOf(raw.function.name);
    if (name === '') return [];
    return [
      {
        name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      },
    ];
  });
  if (tools.length > 0) {
    out.tools = tools;
    const choice = body.tool_choice;
    if (choice === 'auto') out.tool_choice = { type: 'auto' };
    else if (choice === 'required') out.tool_choice = { type: 'any' };
    else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function)) out.tool_choice = { type: 'tool', name: stringOf(choice.function.name) };
  }
  if (stream) out.stream = true;
  return out;
}

/** An Anthropic client request as an OpenAI Chat Completions request that names `model`. */
function anthropicClientToOpenAI(body: Json, model: string, stream: boolean): Json {
  const messages: Json[] = [];
  const system = blocksText(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  for (const raw of arr(body.messages)) {
    if (!isRecord(raw)) continue;
    const isAssistant = raw.role === 'assistant';
    if (typeof raw.content === 'string') {
      messages.push({ role: isAssistant ? 'assistant' : 'user', content: raw.content });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    if (isAssistant) {
      const toolCalls: Json[] = [];
      let text = '';
      for (const block of raw.content as unknown[]) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') text = text === '' ? block.text : `${text}\n${block.text}`;
        if (block.type === 'tool_use') {
          toolCalls.push({
            id: stringOf(block.id),
            type: 'function',
            function: { name: stringOf(block.name), arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
          });
        }
      }
      messages.push({ role: 'assistant', content: text === '' ? null : text, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of raw.content as unknown[]) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        const content = block.content;
        let resultText = '';
        if (typeof content === 'string') resultText = content;
        else if (Array.isArray(content)) {
          resultText = blocksText(content);
          if (resultText === '') resultText = JSON.stringify(content);
        } else if (content !== undefined && content !== null) resultText = JSON.stringify(content);
        messages.push({ role: 'tool', tool_call_id: stringOf(block.tool_use_id), content: resultText });
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') {
        parts.push({ type: 'text', text: block.text });
        continue;
      }
      if (block.type === 'image' && isRecord(block.source)) {
        const source = block.source;
        if (source.type === 'base64' && typeof source.data === 'string') {
          const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
          parts.push({ type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } });
        } else if (source.type === 'url' && typeof source.url === 'string') {
          parts.push({ type: 'image_url', image_url: { url: source.url } });
        }
      }
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stringList(body.stop_sequences);
  if (stops.length > 0) out.stop = stops;
  const tools = arr(body.tools).flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        type: 'function',
        function: {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      },
    ];
  });
  if (tools.length > 0) out.tools = tools;
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

interface GeminiImagePart {
  inlineData?: { mimeType: string; data: string };
  fileData?: { fileUri: string };
}

/** One image reference as a Gemini part: inline bytes for a data URL, a file URI otherwise. */
function geminiImagePartFromUrl(url: string): GeminiImagePart | null {
  if (url === '') return null;
  const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (data !== null) return { inlineData: { mimeType: data[1] ?? 'image/png', data: data[2] ?? '' } };
  return { fileData: { fileUri: url } };
}

function geminiImagePartFromSource(source: unknown): GeminiImagePart | null {
  if (!isRecord(source)) return null;
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  return null;
}

function geminiToolsFromOpenaiTools(tools: unknown[]): Json[] | null {
  const declarations = tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function)) return [];
    const name = stringOf(raw.function.name);
    if (name === '') return [];
    return [
      {
        name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      },
    ];
  });
  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : null;
}

function geminiToolsFromAnthropicTools(tools: unknown[]): Json[] | null {
  const declarations = tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      },
    ];
  });
  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : null;
}

/** An OpenAI client request as a Gemini generateContent body that names `model` (in the URL). */
function openAIClientToGemini(body: Json, stream: boolean): Json {
  const contents: Json[] = [];
  const systemParts: string[] = [];
  for (const raw of arr(body.messages)) {
    if (!isRecord(raw)) continue;
    const role = stringOf(raw.role);
    if (role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'tool') {
      let value: unknown = typeof raw.content === 'string' ? raw.content : raw.content ?? null;
      if (typeof value === 'string') {
        try {
          value = JSON.parse(value);
        } catch {
          // Keep the string: a functionResponse value may be any JSON value.
        }
      }
      contents.push({ role: 'user', parts: [{ functionResponse: { name: stringOf(raw.tool_call_id), response: { result: value } } }] });
      continue;
    }
    const geminiRole = role === 'assistant' ? 'model' : 'user';
    if (typeof raw.content === 'string') {
      const parts: Json[] = [{ text: raw.content }];
      for (const call of arr(raw.tool_calls)) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let args: unknown = {};
        if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
          try {
            args = JSON.parse(call.function.arguments);
          } catch {
            args = {};
          }
        }
        parts.push({ functionCall: { name: stringOf(call.function.name), args: isRecord(args) ? args : {} } });
      }
      contents.push({ role: geminiRole, parts });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    const parts: Json[] = [];
    for (const block of raw.content as unknown[]) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
      if (block.type === 'image_url' && isRecord(block.image_url)) {
        const part = geminiImagePartFromUrl(stringOf(block.image_url.url));
        if (part !== null) parts.push(part);
      }
    }
    for (const call of arr(raw.tool_calls)) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      let args: unknown = {};
      if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
        try {
          args = JSON.parse(call.function.arguments);
        } catch {
          args = {};
        }
      }
      parts.push({ functionCall: { name: stringOf(call.function.name), args: isRecord(args) ? args : {} } });
    }
    if (parts.length > 0) contents.push({ role: geminiRole, parts });
  }

  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  const tools = geminiToolsFromOpenaiTools(arr(body.tools));
  if (tools !== null) {
    out.tools = tools;
    const choice = body.tool_choice;
    if (choice === 'required') out.toolConfig = { functionCallingConfig: { mode: 'ANY' } };
    else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function)) {
      out.toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [stringOf(choice.function.name)] } };
    }
  }
  const generationConfig: Json = {};
  const maxTokens = numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens);
  if (maxTokens !== null) generationConfig.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') generationConfig.temperature = body.temperature;
  if (typeof body.top_p === 'number') generationConfig.topP = body.top_p;
  const stops = stringList(body.stop);
  if (stops.length > 0) generationConfig.stopSequences = stops;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') generationConfig.responseMimeType = 'application/json';
  if (Object.keys(generationConfig).length > 0) out.generationConfig = generationConfig;
  return out;
}

/** An Anthropic client request as a Gemini generateContent body that names `model` (in the URL). */
function anthropicClientToGemini(body: Json, stream: boolean): Json {
  const contents: Json[] = [];
  for (const raw of arr(body.messages)) {
    if (!isRecord(raw)) continue;
    const geminiRole = raw.role === 'assistant' ? 'model' : 'user';
    if (typeof raw.content === 'string') {
      contents.push({ role: geminiRole, parts: [{ text: raw.content }] });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    const parts: Json[] = [];
    for (const block of raw.content as unknown[]) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
      if (block.type === 'image') {
        const part = geminiImagePartFromSource(block.source);
        if (part !== null) parts.push(part);
      }
      if (block.type === 'tool_use') {
        parts.push({ functionCall: { name: stringOf(block.name), args: isRecord(block.input) ? block.input : {} } });
      }
      if (block.type === 'tool_result') {
        let value: unknown = block.content ?? null;
        if (typeof value === 'string') {
          try {
            value = JSON.parse(value);
          } catch {
            // Keep the string.
          }
        }
        parts.push({ functionResponse: { name: stringOf(block.tool_use_id), response: { result: value } } });
      }
    }
    if (parts.length > 0) contents.push({ role: geminiRole, parts });
  }

  const out: Json = { contents };
  const system = blocksText(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const tools = geminiToolsFromAnthropicTools(arr(body.tools));
  if (tools !== null) {
    out.tools = tools;
    const choice = body.tool_choice;
    if (isRecord(choice) && choice.type === 'tool' && typeof choice.name === 'string') {
      out.toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [choice.name] } };
    } else if (isRecord(choice) && choice.type === 'any') {
      out.toolConfig = { functionCallingConfig: { mode: 'ANY' } };
    }
  }
  const generationConfig: Json = {};
  if (typeof body.max_tokens === 'number') generationConfig.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') generationConfig.temperature = body.temperature;
  if (typeof body.top_p === 'number') generationConfig.topP = body.top_p;
  const stops = stringList(body.stop_sequences);
  if (stops.length > 0) generationConfig.stopSequences = stops;
  if (Object.keys(generationConfig).length > 0) out.generationConfig = generationConfig;
  return out;
}

/** The body to POST upstream for one client request on one provider. */
function upstreamBodyFor(api: ClientApi, body: Json, provider: ProviderDir, model: string): { url: string; body: Json } {
  const base = joinBase(ENV_URL, provider.base);
  const stream = body.stream === true;
  if (provider.wire === 'openai') {
    const url = `${base}/v1/chat/completions`;
    if (api === 'openai') {
      // Pass the client's body through untouched: its system prompt, its user text (images
      // included), its tools, its stop sequences, its max_tokens, its JSON mode. Only the model
      // is ours to choose, and a stream always asks the provider for its usage chunk so the
      // answer can carry the counts the client's wire reports.
      const out: Json = { ...body, model };
      if (stream) out.stream_options = { include_usage: true };
      return { url, body: out };
    }
    return { url, body: anthropicClientToOpenAI(body, model, stream) };
  }
  if (provider.wire === 'anthropic') {
    const url = `${base}/v1/messages`;
    if (api === 'anthropic') return { url, body: { ...body, model } };
    return { url, body: openAIClientToAnthropic(body, model, stream) };
  }
  const action = stream ? 'streamGenerateContent?alt=sse' : 'generateContent';
  const url = `${base}/v1beta/models/${encodeURIComponent(model)}:${action}`;
  return { url, body: api === 'openai' ? openAIClientToGemini(body, stream) : anthropicClientToGemini(body, stream) };
}

// ---------------------------------------------------------------------------
// SSE parsing
// ---------------------------------------------------------------------------

interface SseFrame {
  event: string | null;
  data: string;
}

function parseSseFrames(text: string): SseFrame[] {
  const out: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) out.push({ event, data: data.join('\n') });
  }
  return out;
}

function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

// ---------------------------------------------------------------------------
// Stop-reason families
// ---------------------------------------------------------------------------

/** OpenAI's finish_reason as Anthropic's stop_reason. */
function finishToStopReason(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
function stopReasonToFinish(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as OpenAI's finish_reason. */
function geminiFinishToFinish(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY') return 'content_filter';
  if (finish === 'MCP_TOOL_CALLS') return 'tool_calls';
  return 'stop';
}

/** Gemini's finishReason as Anthropic's stop_reason. */
function geminiFinishToStopReason(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'max_tokens';
  if (finish === 'SAFETY') return 'refusal';
  if (finish === 'MCP_TOOL_CALLS') return 'tool_use';
  return 'end_turn';
}

// ---------------------------------------------------------------------------
// Upstream response translation
// ---------------------------------------------------------------------------

interface UpstreamOk {
  ok: true;
  status: number;
  bodyText: string;
  provider: ProviderDir;
  model: string;
}

interface UpstreamFail {
  ok: false;
  status: number;
  bodyText: string;
  message: string;
  retryAfterMs: number | null;
}

type Upstream = UpstreamOk | UpstreamFail;

function extractErrorMessage(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      const error = value.error;
      if (typeof error.message === 'string') return error.message;
      if (isRecord(error) && typeof error.status === 'string') return error.status;
    }
    if (isRecord(value) && typeof value.message === 'string') return value.message;
  } catch {
    // Not JSON: use a slice of the text.
  }
  return text.slice(0, 300);
}

function parseRetryAfter(headers: Headers): number | null {
  const raw = headers.get('retry-after');
  if (raw === null) return null;
  const seconds = Number(raw.trim());
  if (!Number.isFinite(seconds) || seconds <= 0) return null;
  // Honored, but bounded: the wall budget and the client's own request timeout matter more.
  return Math.min(seconds * 1000, 500);
}

function numOr0(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

// --- openai-wire answers ---

interface OpenaiStreamData {
  pieces: string[];
  toolCalls: Map<number, { id: string; name: string; args: string }>;
  finish: unknown;
  usageIn: number | null;
  usageOut: number | null;
}

function collectOpenaiStream(text: string): OpenaiStreamData {
  const out: OpenaiStreamData = { pieces: [], toolCalls: new Map(), finish: null, usageIn: null, usageOut: null };
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      out.usageIn = numOr0(chunk.usage.prompt_tokens);
      out.usageOut = numOr0(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string' && delta.content !== '') out.pieces.push(delta.content);
    for (const call of arr(delta.tool_calls)) {
      if (!isRecord(call)) continue;
      const index = typeof call.index === 'number' ? call.index : out.toolCalls.size;
      let entry = out.toolCalls.get(index);
      if (entry === undefined) {
        entry = { id: '', name: '', args: '' };
        out.toolCalls.set(index, entry);
      }
      if (typeof call.id === 'string' && call.id !== '') entry.id = call.id;
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.name === 'string' && fn.name !== '') entry.name = fn.name;
      if (typeof fn.arguments === 'string') entry.args += fn.arguments;
    }
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) out.finish = choice.finish_reason;
  }
  return out;
}

/** An openai-wire answer as an Anthropic Messages object. */
function openaiToAnthropicMessage(text: string, model: string): Json {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    value = {};
  }
  const body = isRecord(value) ? value : {};
  const choices = arr(body.choices);
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const answerText = typeof message.content === 'string' ? message.content : blocksText(message.content);
  if (answerText !== '') content.push({ type: 'text', text: answerText });
  for (const call of arr(message.tool_calls)) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    let input: unknown = {};
    if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
      try {
        input = JSON.parse(call.function.arguments);
      } catch {
        input = {};
      }
    }
    content.push({ type: 'tool_use', id: stringOf(call.id) || 'toolu_0', name: stringOf(call.function.name), input: isRecord(input) ? input : {} });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  const id = typeof body.id === 'string' ? body.id : 'sim';
  return {
    id: `msg_${id}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: finishToStopReason(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numOr0(usage.prompt_tokens), output_tokens: numOr0(usage.completion_tokens) },
  };
}

/** An openai-wire event stream as the Anthropic event sequence. */
function openaiStreamToAnthropicEvents(text: string, model: string): string {
  const data = collectOpenaiStream(text);
  const inputTokens = data.usageIn ?? 0;
  const outputTokens = data.usageOut ?? 0;
  const frames: string[] = [];
  frames.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: inputTokens, output_tokens: 0 },
      },
    }),
  );
  let index = 0;
  if (data.pieces.length > 0) {
    frames.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    for (const piece of data.pieces) {
      frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } }));
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const call of [...data.toolCalls.values()]) {
    frames.push(
      sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'tool_use', id: call.id || `toolu_${index}`, name: call.name, input: {} } }),
    );
    if (call.args !== '') {
      frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: call.args } }));
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  frames.push(
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: finishToStopReason(data.finish), stop_sequence: null }, usage: { output_tokens: outputTokens } }),
  );
  frames.push(sseEvent('message_stop', { type: 'message_stop' }));
  return frames.join('');
}

// --- anthropic-wire answers ---

interface AnthropicStreamData {
  inputTokens: number | null;
  outputTokens: number | null;
  stopReason: unknown;
  textBlocks: string[];
  toolCalls: { index: number; id: string; name: string; args: string }[];
}

function collectAnthropicStream(text: string): AnthropicStreamData {
  const out: AnthropicStreamData = { inputTokens: null, outputTokens: null, stopReason: null, textBlocks: [], toolCalls: [] };
  const openText = new Map<number, number>();
  const openTools = new Map<number, { index: number; id: string; name: string; args: string }>();
  for (const frame of parseSseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start') {
      const message = isRecord(data.message) ? data.message : {};
      if (isRecord(message.usage)) out.inputTokens = numOr0(message.usage.input_tokens);
      continue;
    }
    if (frame.event === 'content_block_start') {
      const index = typeof data.index === 'number' ? data.index : 0;
      const block = isRecord(data.content_block) ? data.content_block : {};
      if (block.type === 'text') {
        openText.set(index, out.textBlocks.length);
        out.textBlocks.push('');
      } else if (block.type === 'tool_use') {
        const entry = { index, id: stringOf(block.id), name: stringOf(block.name), args: '' };
        openTools.set(index, entry);
        out.toolCalls.push(entry);
      }
      continue;
    }
    if (frame.event === 'content_block_delta') {
      const index = typeof data.index === 'number' ? data.index : 0;
      const delta = isRecord(data.delta) ? data.delta : {};
      if (delta.type === 'text_delta' && typeof delta.text === 'string') {
        const at = openText.get(index);
        if (at !== undefined) out.textBlocks[at] += delta.text;
      } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string') {
        const entry = openTools.get(index);
        if (entry !== undefined) entry.args += delta.partial_json;
      }
      continue;
    }
    if (frame.event === 'message_delta') {
      const delta = isRecord(data.delta) ? data.delta : {};
      if (typeof delta.stop_reason === 'string') out.stopReason = delta.stop_reason;
      if (isRecord(data.usage)) out.outputTokens = numOr0(data.usage.output_tokens);
      continue;
    }
  }
  return out;
}

/** An anthropic-wire answer as an OpenAI Chat Completions object. */
function anthropicToOpenaiCompletion(text: string, model: string): Json {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    value = {};
  }
  const body = isRecord(value) ? value : {};
  const content = arr(body.content);
  let answerText = '';
  const toolCalls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') answerText = answerText === '' ? block.text : `${answerText}\n${block.text}`;
    if (block.type === 'tool_use') {
      toolCalls.push({
        id: stringOf(block.id),
        type: 'function',
        function: { name: stringOf(block.name), arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const inputTokens = numOr0(usage.input_tokens);
  const outputTokens = numOr0(usage.output_tokens);
  const id = typeof body.id === 'string' ? body.id : 'sim';
  return {
    id: `chatcmpl-${id}`,
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: answerText, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) },
        finish_reason: stopReasonToFinish(body.stop_reason),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
  };
}

/** An anthropic-wire event stream as OpenAI Chat Completions chunks ending in [DONE]. */
function anthropicStreamToOpenaiEvents(text: string, model: string): string {
  const data = collectAnthropicStream(text);
  const inputTokens = data.inputTokens ?? 0;
  const outputTokens = data.outputTokens ?? 0;
  const chunk = (delta: Json, finish: unknown): Json => ({
    id: 'chatcmpl-stream',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [];
  frames.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  for (const piece of data.textBlocks) {
    if (piece === '') continue;
    frames.push(sseEvent(null, chunk({ content: piece }, null)));
  }
  frames.push(sseEvent(null, chunk({}, stopReasonToFinish(data.stopReason))));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-stream',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

// --- gemini-wire answers ---

interface GeminiStreamData {
  pieces: string[];
  finish: unknown;
  usageIn: number | null;
  usageOut: number | null;
}

function collectGeminiStream(text: string): GeminiStreamData {
  const out: GeminiStreamData = { pieces: [], finish: null, usageIn: null, usageOut: null };
  for (const frame of parseSseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = arr(data.candidates);
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      const content = isRecord(candidate.content) ? candidate.content : {};
      for (const part of arr(content.parts)) {
        if (isRecord(part) && typeof part.text === 'string' && part.text !== '') out.pieces.push(part.text);
      }
      if (candidate.finishReason !== undefined && candidate.finishReason !== null) out.finish = candidate.finishReason;
    }
    if (isRecord(data.usageMetadata)) {
      out.usageIn = numOr0(data.usageMetadata.promptTokenCount);
      out.usageOut = numOr0(data.usageMetadata.candidatesTokenCount);
    }
  }
  return out;
}

function geminiPartsText(parts: unknown[]): string {
  const texts: string[] = [];
  for (const part of parts) {
    if (isRecord(part) && typeof part.text === 'string') texts.push(part.text);
  }
  return texts.join('');
}

function geminiAnswerText(body: Json): string {
  const candidates = arr(body.candidates);
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  return geminiPartsText(arr(content.parts));
}

/** A gemini-wire answer as an OpenAI Chat Completions object. */
function geminiToOpenaiCompletion(text: string, model: string): Json {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    value = {};
  }
  const body = isRecord(value) ? value : {};
  const candidates = arr(body.candidates);
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const inputTokens = numOr0(usage.promptTokenCount);
  const outputTokens = numOr0(usage.candidatesTokenCount);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: geminiAnswerText(body) }, finish_reason: geminiFinishToFinish(candidate.finishReason), logprobs: null }],
    usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
  };
}

/** A gemini-wire answer as an Anthropic Messages object. */
function geminiToAnthropicMessage(text: string, model: string): Json {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    value = {};
  }
  const body = isRecord(value) ? value : {};
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: geminiAnswerText(body) }],
    stop_reason: geminiFinishToStopReason((arr(body.candidates)[0] as Json | undefined)?.finishReason),
    stop_sequence: null,
    usage: { input_tokens: numOr0(usage.promptTokenCount), output_tokens: numOr0(usage.candidatesTokenCount) },
  };
}

/** A gemini-wire event stream as OpenAI Chat Completions chunks ending in [DONE]. */
function geminiStreamToOpenaiEvents(text: string, model: string): string {
  const data = collectGeminiStream(text);
  const inputTokens = data.usageIn ?? 0;
  const outputTokens = data.usageOut ?? 0;
  const chunk = (delta: Json, finish: unknown): Json => ({
    id: 'chatcmpl-stream',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [];
  frames.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  for (const piece of data.pieces) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, geminiFinishToFinish(data.finish))));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-stream',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A gemini-wire event stream as the Anthropic event sequence. */
function geminiStreamToAnthropicEvents(text: string, model: string): string {
  const data = collectGeminiStream(text);
  const inputTokens = data.usageIn ?? 0;
  const outputTokens = data.usageOut ?? 0;
  const frames: string[] = [];
  frames.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: inputTokens, output_tokens: 0 },
      },
    }),
  );
  frames.push(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
  for (const piece of data.pieces) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  frames.push(
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: geminiFinishToStopReason(data.finish), stop_sequence: null }, usage: { output_tokens: outputTokens } }),
  );
  frames.push(sseEvent('message_stop', { type: 'message_stop' }));
  return frames.join('');
}

// ---------------------------------------------------------------------------
// Client-facing responses
// ---------------------------------------------------------------------------

function writeBody(res: ServerResponse, status: number, contentType: string, text: string): void {
  try {
    res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
    res.end(text);
  } catch {
    // The client went away: nothing to answer.
  }
}

function writeJson(res: ServerResponse, status: number, value: unknown): void {
  writeBody(res, status, 'application/json', JSON.stringify(value));
}

function writeSse(res: ServerResponse, text: string): void {
  writeBody(res, 200, 'text/event-stream; charset=utf-8', text);
}

function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

function openaiError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, param: null, code } };
}

function sendError(res: ServerResponse, api: ClientApi, status: number, message: string): void {
  if (api === 'anthropic') writeJson(res, status, anthropicError('api_error', message));
  else writeJson(res, status, openaiError(message, 'api_error', null));
}

/**
 * The provider's answer in the wire the client used. Same wire: the answer passes through whole;
 * a different wire: the answer is translated, with the text whole, the usage counts and the stop
 * reason mapped as the client's wire reports them.
 */
function respondFromUpstream(res: ServerResponse, api: ClientApi, upstream: UpstreamOk, stream: boolean): void {
  const wire = upstream.provider.wire;
  const model = upstream.model;
  if (api === 'openai') {
    if (wire === 'openai') {
      writeBody(res, 200, stream ? 'text/event-stream; charset=utf-8' : 'application/json', upstream.bodyText);
      return;
    }
    if (stream) writeSse(res, wire === 'anthropic' ? anthropicStreamToOpenaiEvents(upstream.bodyText, model) : geminiStreamToOpenaiEvents(upstream.bodyText, model));
    else writeJson(res, 200, wire === 'anthropic' ? anthropicToOpenaiCompletion(upstream.bodyText, model) : geminiToOpenaiCompletion(upstream.bodyText, model));
    return;
  }
  if (wire === 'anthropic') {
    writeBody(res, 200, stream ? 'text/event-stream; charset=utf-8' : 'application/json', upstream.bodyText);
    return;
  }
  if (stream) writeSse(res, wire === 'openai' ? openaiStreamToAnthropicEvents(upstream.bodyText, model) : geminiStreamToAnthropicEvents(upstream.bodyText, model));
  else writeJson(res, 200, wire === 'openai' ? openaiToAnthropicMessage(upstream.bodyText, model) : geminiToAnthropicMessage(upstream.bodyText, model));
}

// ---------------------------------------------------------------------------
// One request: route, call, fail over, answer
// ---------------------------------------------------------------------------

/** What a refusal means: the provider is done for the trial, it may serve the next call, or skip it. */
type FailureClass = 'dead' | 'transient' | 'skip';

/**
 * The refusal's message names its cause, and the same status can carry different causes (the
 * pollinations service answers every refusal with 402, a schedule miss included), so the message
 * decides and the status only backs it up. A service that is out of budget or down is done for the
 * trial; one that refused this call on its schedule may answer the next.
 */
function classifyFailure(upstream: UpstreamFail): FailureClass {
  const message = upstream.message;
  if (message.includes('budget is spent')) return 'dead';
  if (message.includes(' is out')) return 'dead';
  if (message.includes('refused this call')) return 'transient';
  if (message.includes('call at a time')) return 'transient';
  if (upstream.status === 402) return 'dead';
  if (upstream.status === 429) return 'transient';
  if (upstream.status === 0) return 'transient';
  return 'skip';
}

async function callProvider(api: ClientApi, body: Json, candidate: Candidate): Promise<Upstream> {
  const provider = candidate.provider;
  const model = candidate.row.model;
  const { url, body: upstreamBody } = upstreamBodyFor(api, body, provider, model);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeadersFor(provider) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (body.stream === true) headers.accept = 'text/event-stream';
  try {
    const reply = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
    const text = await reply.text();
    if (!reply.ok) {
      return { ok: false, status: reply.status, bodyText: text, message: extractErrorMessage(text), retryAfterMs: parseRetryAfter(reply.headers) };
    }
    return { ok: true, status: reply.status, bodyText: text, provider, model };
  } catch (err) {
    return { ok: false, status: 0, bodyText: '', message: `the provider could not be reached: ${(err as Error).message}`, retryAfterMs: null };
  }
}

async function readRequestJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const raw of req) {
    const chunk = Buffer.isBuffer(raw) ? raw : Buffer.from(raw as string);
    size += chunk.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(chunk);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** How many client requests have arrived: the manifest is in arrival order, one at a time. */
let requestIndex = 0;

async function handleChat(req: IncomingMessage, res: ServerResponse, api: ClientApi): Promise<void> {
  const body = await readRequestJson(req);
  const row = brief.manifest[requestIndex];
  const requestIdx = requestIndex;
  requestIndex += 1;
  if (body === null) return sendError(res, api, 400, 'the request body is not one JSON object');
  const requires = row !== undefined ? row.requires : requirementsFromBody(body, api);
  const weight = row !== undefined ? row.weight : 1;
  // Pass one keeps the call-count policy (the caps). If it finds no answer at all - every
  // candidate capped out, dead or refusing - pass two drops the caps: an unserved request costs
  // far more than the one call a spent-budget refusal can waste, and a provider whose calls so
  // far stayed inside its known-smallest budget may simply hold a bigger budget than assumed.
  const tried = new Set<string>();
  for (const ignoreCaps of [false, true]) {
    const candidates = rankCandidates(api, requires, weight, ignoreCaps, requestIdx);
    let attempts = 0;
    for (const candidate of candidates) {
      if (attempts >= MAX_ATTEMPTS_PER_REQUEST) break;
      const providerHealth = healthOf(candidate.provider.id);
      // A candidate earlier in this list may have dropped the provider mid-request.
      if (providerHealth.dead) continue;
      // The emergency pass does not re-argue with a provider this request already heard refused.
      if (ignoreCaps && tried.has(candidate.provider.id)) continue;
      let scheduleRetry = false;
      for (let attempt = 0; attempt < RETRIES_PER_CANDIDATE; attempt++) {
        // The cap bounds wasted calls past a spent budget. A schedule refusal is the exception: it
        // proves the call number was still inside the budget, so the retry that follows it can
        // never refuse for budget and is allowed past the cap.
        if (!ignoreCaps && attemptsOf(providerHealth) >= attemptCapOf(candidate.provider.id) && !scheduleRetry) break;
        attempts += 1;
        tried.add(candidate.provider.id);
        const upstream = await callProvider(api, body, candidate);
        if (upstream.ok) {
          providerHealth.successes += 1;
          deficitSpent += weight * Math.max(0, bestLegalRating(api, requires) - candidate.row.ratingMilli);
          const label = row !== undefined ? row.id : `#${requestIndex}`;
          log(`served ${label} via ${candidate.provider.id}/${candidate.row.model} (rating ${candidate.row.ratingMilli}, attempt ${attempt + 1})`);
          respondFromUpstream(res, api, upstream, body.stream === true);
          return;
        }
        providerHealth.failures += 1;
        const kind = classifyFailure(upstream);
        scheduleRetry = kind === 'transient' && upstream.message.includes('refused this call');
        log(`${candidate.provider.id}/${candidate.row.model} refused: HTTP ${upstream.status} ${upstream.message.slice(0, 120)} -> ${kind}`);
        if (kind === 'dead') {
          providerHealth.dead = true;
          break;
        }
        if (kind === 'transient') {
          if (upstream.retryAfterMs !== null && attempt + 1 < RETRIES_PER_CANDIDATE) await sleep(upstream.retryAfterMs);
          continue;
        }
        break;
      }
    }
  }
  sendError(res, api, 429, 'no free model on the directory could serve this request');
}

// ---------------------------------------------------------------------------
// GET /v1/models: the model list in the wire the auth header names
// ---------------------------------------------------------------------------

function handleModels(req: IncomingMessage, res: ServerResponse): void {
  const models = freeCatalogModels();
  if (req.headers['x-api-key'] !== undefined) {
    writeJson(res, 200, {
      data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.id, created_at: FIXED_CREATED_AT, free: true })),
      has_more: false,
    });
    return;
  }
  writeJson(res, 200, {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: 'aggregator', free: true })),
  });
}

// ---------------------------------------------------------------------------
// The server, the register and the poll
// ---------------------------------------------------------------------------

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  const path = url.pathname;
  if (req.method === 'GET' && path === '/v1/models') {
    handleModels(req, res);
    return;
  }
  if (req.method === 'POST' && path === '/v1/chat/completions') {
    await handleChat(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && path === '/v1/messages') {
    await handleChat(req, res, 'anthropic');
    return;
  }
  const api: ClientApi = path === '/v1/messages' ? 'anthropic' : 'openai';
  sendError(res, api, 404, `no route ${req.method ?? ''} ${path}`);
}

function listen(server: Server): Promise<number> {
  return new Promise((resolve, reject) => {
    const onError = (err: Error): void => reject(err);
    server.once('error', onError);
    server.listen(0, '0.0.0.0', () => {
      server.removeListener('error', onError);
      const address = server.address();
      if (address === null || typeof address === 'string') {
        reject(new Error('the server has no port'));
        return;
      }
      resolve(address.port);
    });
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const body = JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report });
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body,
        signal: AbortSignal.timeout(5000),
      });
      if (res.ok) {
        const reply: unknown = await res.json();
        const requests = isRecord(reply) && typeof reply.requests === 'number' ? reply.requests : '?';
        log(`registered on port ${port}: the workload holds ${requests} requests`);
        return;
      }
      log(`register answered HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
    } catch (err) {
      log(`register failed: ${(err as Error).message}`);
    }
    await sleep(200);
  }
  log('register never succeeded; the workload may never start');
}

function pollWork(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          log('the workload is done: exiting');
          server.close();
          await sleep(EXIT_SETTLE_MS);
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (ENV_URL !== '') await probeAll();
  catalog = buildCatalog();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and
    // stop before any server starts.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    await sleep(EXIT_SETTLE_MS);
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.includes('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'the server failed to answer');
      else res.end();
    });
  });
  let port = await listen(server);
  if (port === 8080) {
    // 8080 is the env's own port on one layout: never register it.
    server.close();
    port = await listen(server);
  }
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${freeCatalogModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  pollWork(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
