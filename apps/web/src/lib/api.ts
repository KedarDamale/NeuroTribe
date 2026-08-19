/**
 * Typed API client.
 *
 * Server-side rendering talks to the API container directly; the browser goes
 * through the same origin (`/api`) so there is no CORS surface and no hard-coded
 * host in the shipped bundle.
 */

const SERVER_BASE =
  import.meta.env.API_INTERNAL_BASE ??
  process.env.API_INTERNAL_BASE ??
  'http://localhost:8000/api';

const CLIENT_BASE = import.meta.env.PUBLIC_API_BASE ?? '/api';

export const apiBase = (): string =>
  typeof window === 'undefined' ? SERVER_BASE : CLIENT_BASE;

/** A browser-reachable URL safe to render into server-side HTML links. */
export const publicApiBase = (): string => CLIENT_BASE;

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly path: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${apiBase()}${path}`;
  let response: Response;
  try {
    response = await fetch(url, {
      ...init,
      headers: { Accept: 'application/json', ...(init?.headers ?? {}) },
    });
  } catch (cause) {
    throw new ApiError(
      `Cannot reach the NeuroTRIBE API at ${url}. Is the api service running?`,
      0,
      path,
    );
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body?.detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail, response.status, path);
  }
  return (await response.json()) as T;
}

/** GET that degrades to a fallback instead of breaking the whole page. */
export async function safeGet<T>(path: string, fallback: T): Promise<T> {
  try {
    return await request<T>(path);
  } catch (error) {
    if (typeof console !== 'undefined') {
      console.warn(`[neurotribe] ${path} unavailable:`, (error as Error).message);
    }
    return fallback;
  }
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, {
      method: 'POST',
      headers: body ? { 'Content-Type': 'application/json' } : {},
      body: body ? JSON.stringify(body) : undefined,
    }),
  /** Fetch a packed binary buffer (vertex maps, surface geometry). */
  buffer: async (path: string): Promise<{ data: ArrayBuffer; headers: Headers }> => {
    const response = await fetch(`${apiBase()}${path}`);
    if (!response.ok) {
      throw new ApiError(response.statusText, response.status, path);
    }
    return { data: await response.arrayBuffer(), headers: response.headers };
  },
};

// ------------------------------------------------------------------ types

export type StageState =
  | 'PENDING' | 'RUNNING' | 'DONE' | 'WAITING_EXTERNAL'
  | 'FAILED_RETRYABLE' | 'FAILED_FINAL' | 'SKIPPED' | 'BLOCKED' | 'PARTIAL';

export interface Stage {
  key: string;
  label: string;
  phase: number;
  state: StageState;
  detail: string | null;
  progress: number;
  attempts: number;
  max_attempts: number;
  last_error: string | null;
  depends_on: string[];
  description?: string;
  result?: Record<string, unknown>;
}

export interface Blocker {
  id: string;
  kind: string;
  severity: 'EXTERNAL' | 'ACTIONABLE' | 'INFO';
  title: string;
  description: string;
  required_action: string | null;
  reference_url: string | null;
  blocks_stages: string[];
  context: Record<string, unknown>;
}

export interface SystemProbe {
  hostname: string;
  platform: string;
  python_version: string;
  cpu_count: number;
  ram_gb: number | null;
  free_disk_gb: number | null;
  total_disk_gb: number | null;
  gpu_name: string | null;
  vram_gb: number | null;
  cuda_available: boolean;
  docker_available: boolean;
  docker_version: string | null;
  docker_memory_gb: number | null;
  ffmpeg_available: boolean;
  tribe_available: boolean;
  freesurfer_license: boolean;
  ready: boolean;
  warnings: string[];
  blockers: string[];
}

export interface Dashboard {
  disclaimer: string;
  profile: string;
  analysis_config_hash: string;
  cards: {
    dataset: string;
    subjects_indexed: number;
    movie_fmri_subjects: number;
    movie_scans: number;
    adhd_labels: string;
    n_phenotype_subjects: number;
    tribe_model: string;
    stimulus: string;
    pipeline: string;
    overall_progress: number;
    n_valid_comparisons: number;
    n_assets: number;
  };
  pipeline: {
    groups: { name: string; stages: Stage[] }[];
    by_state: Record<string, number>;
    n_stages: number;
  };
  blockers: Blocker[];
  system: SystemProbe | null;
  cohort: {
    n_case: number; n_control: number; n_excluded: number;
    warnings: string[]; movie: string;
  } | null;
  latest_group_run: {
    id: string; name: string; tier: string; status: string;
    sanity_passed: boolean; summary: Record<string, unknown>;
  } | null;
}

export interface GroupResultRow {
  unit_type: string;
  unit_name: string;
  unit_index: number | null;
  network: string | null;
  metric: string;
  mean_case: number | null;
  mean_control: number | null;
  sd_case: number | null;
  sd_control: number | null;
  beta_adhd: number | null;
  se_adhd: number | null;
  t_stat: number | null;
  p_value: number | null;
  q_value: number | null;
  effect_size: number | null;
  ci_low: number | null;
  ci_high: number | null;
  n_case: number | null;
  n_control: number | null;
}

export interface PeakWindow {
  rank: number;
  start_sec: number;
  end_sec: number;
  start_label: string;
  end_label: string;
  deviation: number;
  coverage: number;
}

export interface SubjectSummary {
  external_id: string;
  site: string | null;
  age: number | null;
  sex: string | null;
  has_phenotype: boolean;
  has_movie_bold: boolean;
  movies: string[];
  diagnoses: { label: string; certainty: string; is_adhd: boolean }[];
  comparison: {
    id: string;
    valid: boolean;
    global_agreement_r: number | null;
    global_mad: number | null;
    usable_frame_fraction: number | null;
    is_approximate: boolean;
  } | null;
}

// ------------------------------------------------------------------ format

export const fmt = {
  num(value: number | null | undefined, digits = 3): string {
    if (value === null || value === undefined || Number.isNaN(value)) return '—';
    if (value !== 0 && Math.abs(value) < 0.001) return value.toExponential(1);
    return value.toFixed(digits);
  },
  pct(value: number | null | undefined, digits = 0): string {
    if (value === null || value === undefined || Number.isNaN(value)) return '—';
    return `${(value * 100).toFixed(digits)}%`;
  },
  int(value: number | null | undefined): string {
    if (value === null || value === undefined) return '—';
    return value.toLocaleString();
  },
  bytes(value: number | null | undefined): string {
    if (!value) return '—';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let size = value;
    let unit = 0;
    while (size >= 1024 && unit < units.length - 1) {
      size /= 1024;
      unit += 1;
    }
    return `${size.toFixed(size >= 100 || unit === 0 ? 0 : 1)} ${units[unit]}`;
  },
  duration(seconds: number | null | undefined): string {
    if (seconds === null || seconds === undefined) return '—';
    if (seconds < 60) return `${seconds.toFixed(0)}s`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ${Math.round(seconds % 60)}s`;
    return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
  },
  timecode(seconds: number | null | undefined): string {
    if (seconds === null || seconds === undefined) return '—';
    const total = Math.max(0, Math.floor(seconds));
    return `${String(Math.floor(total / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`;
  },
};
