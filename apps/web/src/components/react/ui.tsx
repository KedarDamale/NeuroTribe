/** Shared presentational primitives used across the dashboard islands. */

import type { ReactNode } from 'react';
import { useEffect, useState } from 'react';
import type { StageState } from '../../lib/api';

export const STATE_COLOR: Record<string, string> = {
  DONE: 'var(--color-vital)',
  RUNNING: 'var(--color-signal)',
  PARTIAL: 'var(--color-cyan)',
  WAITING_EXTERNAL: 'var(--color-ember)',
  PENDING: 'var(--fg-faint)',
  BLOCKED: 'var(--fg-faint)',
  SKIPPED: 'var(--fg-faint)',
  FAILED_RETRYABLE: 'var(--color-alarm)',
  FAILED_FINAL: 'var(--color-alarm)',
  PASS: 'var(--color-vital)',
  WARNING: 'var(--color-ember)',
  FAIL: 'var(--color-alarm)',
  UNKNOWN: 'var(--fg-faint)',
  SUCCEEDED: 'var(--color-vital)',
  QUEUED: 'var(--fg-faint)',
  CACHED: 'var(--color-cyan)',
  CANCELLED: 'var(--fg-faint)',
  EXTERNAL: 'var(--color-ember)',
  ACTIONABLE: 'var(--color-signal)',
  INFO: 'var(--fg-muted)',
};

export const STATE_LABEL: Partial<Record<StageState, string>> = {
  WAITING_EXTERNAL: 'Waiting · external',
  FAILED_RETRYABLE: 'Failed · will retry',
  FAILED_FINAL: 'Failed',
  BLOCKED: 'Blocked upstream',
  PARTIAL: 'In progress',
};

export function Badge({
  state,
  children,
  dot = true,
}: {
  state: string;
  children?: ReactNode;
  dot?: boolean;
}) {
  const color = STATE_COLOR[state] ?? 'var(--fg-muted)';
  return (
    <span
      className="inline-flex shrink-0 items-center gap-1.5 rounded-full px-2 py-0.5 text-[0.68rem] font-semibold"
      style={{
        color,
        background: `color-mix(in oklab, ${color} 15%, transparent)`,
        border: `1px solid color-mix(in oklab, ${color} 32%, transparent)`,
      }}
    >
      {dot && (
        <span
          className={`inline-block h-1.5 w-1.5 rounded-full ${state === 'RUNNING' ? 'animate-pulse-soft' : ''}`}
          style={{ background: color }}
        />
      )}
      {children ?? STATE_LABEL[state as StageState] ?? state.replaceAll('_', ' ')}
    </span>
  );
}

export function Card({
  label,
  value,
  hint,
  accent,
  children,
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  accent?: string;
  children?: ReactNode;
}) {
  return (
    <div className="panel relative overflow-hidden p-4">
      {accent && (
        <div
          className="absolute inset-x-0 top-0 h-0.5"
          style={{ background: accent }}
          aria-hidden="true"
        />
      )}
      <div
        className="text-[0.66rem] font-semibold uppercase tracking-[0.09em]"
        style={{ color: 'var(--fg-faint)' }}
      >
        {label}
      </div>
      <div className="mt-1.5 text-2xl font-semibold tracking-tight numeric">{value}</div>
      {hint && (
        <div className="mt-1 text-[0.74rem]" style={{ color: 'var(--fg-muted)' }}>
          {hint}
        </div>
      )}
      {children}
    </div>
  );
}

export function Section({
  title,
  description,
  actions,
  children,
}: {
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="animate-rise">
      <div className="mb-3 flex flex-wrap items-end justify-between gap-2">
        <div>
          <h2 className="text-[1.05rem] font-semibold tracking-tight">{title}</h2>
          {description && (
            <p className="mt-0.5 max-w-2xl text-[0.8rem]" style={{ color: 'var(--fg-muted)' }}>
              {description}
            </p>
          )}
        </div>
        {actions}
      </div>
      {children}
    </section>
  );
}

export function Progress({ value, state = 'RUNNING' }: { value: number; state?: string }) {
  const color = STATE_COLOR[state] ?? 'var(--color-signal)';
  const indeterminate = state === 'RUNNING' && value <= 0;
  return (
    <div
      className={`relative h-1 w-full overflow-hidden rounded-full ${indeterminate ? 'sweep' : ''}`}
      style={{ background: 'color-mix(in oklab, var(--fg-faint) 22%, transparent)' }}
      role="progressbar"
      aria-valuenow={Math.round(value * 100)}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      {!indeterminate && (
        <div
          className="h-full rounded-full transition-[width] duration-500"
          style={{ width: `${Math.min(100, Math.max(0, value * 100))}%`, background: color }}
        />
      )}
    </div>
  );
}

export function Empty({ title, body }: { title: string; body?: ReactNode }) {
  return (
    <div className="panel grid place-items-center px-6 py-14 text-center">
      <div className="max-w-md">
        <div className="mb-2 text-2xl opacity-40" aria-hidden="true">
          ◌
        </div>
        <p className="text-sm font-medium">{title}</p>
        {body && (
          <p className="mt-1.5 text-[0.8rem] leading-relaxed" style={{ color: 'var(--fg-muted)' }}>
            {body}
          </p>
        )}
      </div>
    </div>
  );
}

export function Table({
  columns,
  children,
  dense = false,
}: {
  columns: string[];
  children: ReactNode;
  dense?: boolean;
}) {
  return (
    <div className="panel overflow-x-auto">
      <table className="w-full min-w-[42rem] border-collapse text-left">
        <thead>
          <tr>
            {columns.map((column) => (
              <th
                key={column}
                className={`whitespace-nowrap border-b px-3 ${dense ? 'py-1.5' : 'py-2.5'} text-[0.64rem] font-semibold uppercase tracking-[0.07em]`}
                style={{ color: 'var(--fg-faint)' }}
              >
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

export function Td({
  children,
  numeric = false,
  muted = false,
  title,
}: {
  children: ReactNode;
  numeric?: boolean;
  muted?: boolean;
  title?: string;
}) {
  return (
    <td
      title={title}
      className={`whitespace-nowrap border-b px-3 py-2 text-[0.82rem] ${numeric ? 'numeric' : ''}`}
      style={muted ? { color: 'var(--fg-muted)' } : undefined}
    >
      {children}
    </td>
  );
}

/** Poll a loader on an interval; pauses when the tab is hidden. */
export function usePolling<T>(
  loader: () => Promise<T>,
  intervalMs = 5000,
  enabled = true,
): { data: T | null; error: string | null; refresh: () => void } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;

    const run = async () => {
      if (typeof document !== 'undefined' && document.hidden) return;
      try {
        const result = await loader();
        if (!cancelled) {
          setData(result);
          setError(null);
        }
      } catch (cause) {
        if (!cancelled) setError((cause as Error).message);
      }
    };

    void run();
    const timer = setInterval(run, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs, enabled, nonce]);

  return { data, error, refresh: () => setNonce((n) => n + 1) };
}
