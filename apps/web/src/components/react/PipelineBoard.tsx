/**
 * Live pipeline board.
 *
 * Polls the dashboard endpoint so the operator can literally watch the
 * Autopilot work. Colour follows the specification legend:
 *   green = done · blue = running · amber = waiting external
 *   red = failed · grey = blocked downstream
 */

import { useState } from 'react';
import { api, fmt, type Dashboard, type Stage } from '../../lib/api';
import { Badge, Card, Progress, STATE_COLOR, usePolling } from './ui';

interface Props {
  initial: Dashboard;
  pollMs?: number;
}

export default function PipelineBoard({ initial, pollMs = 5000 }: Props) {
  const { data, error, refresh } = usePolling<Dashboard>(
    () => api.get<Dashboard>('/dashboard'),
    pollMs,
  );
  const dashboard = data ?? initial;
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  const cards = dashboard.cards;

  const runTick = async () => {
    setBusy(true);
    setNotice(null);
    try {
      const result = await api.post<{ ran: string[]; outcomes: Record<string, string> }>(
        '/pipeline/tick',
      );
      setNotice(
        result.ran.length
          ? `Advanced: ${result.ran.join(', ')}`
          : 'Nothing is runnable right now — every ready stage is waiting on an external input.',
      );
      refresh();
    } catch (cause) {
      setNotice(`Tick failed: ${(cause as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const retry = async (key: string) => {
    setBusy(true);
    try {
      await api.post(`/pipeline/stages/${key}/retry`);
      refresh();
    } catch (cause) {
      setNotice(`Retry failed: ${(cause as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const pipelineAccent = cards.pipeline.startsWith('BLOCKED')
    ? 'var(--color-ember)'
    : cards.pipeline === 'RUNNING'
      ? 'var(--color-signal)'
      : cards.pipeline === 'COMPLETE'
        ? 'var(--color-vital)'
        : 'var(--fg-faint)';

  return (
    <div className="space-y-6">
      {error && (
        <div
          className="panel px-4 py-3 text-[0.8rem]"
          style={{ color: 'var(--color-ember)' }}
        >
          Live updates paused: {error}
        </div>
      )}

      {/* Headline cards */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <Card label="Dataset" value={cards.dataset} hint={`${fmt.int(cards.n_assets)} assets`} />
        <Card
          label="Subjects indexed"
          value={fmt.int(cards.subjects_indexed)}
          hint={`${fmt.int(cards.movie_fmri_subjects)} with movie fMRI`}
        />
        <Card
          label="ADHD labels"
          value={cards.adhd_labels}
          accent={cards.adhd_labels === 'READY' ? 'var(--color-vital)' : 'var(--color-ember)'}
          hint={
            cards.adhd_labels === 'READY'
              ? `${fmt.int(cards.n_phenotype_subjects)} phenotyped`
              : 'DUA-controlled'
          }
        />
        <Card
          label="TRIBE model"
          value={cards.tribe_model}
          accent={cards.tribe_model === 'MOCK' ? 'var(--color-ember)' : undefined}
          hint={cards.tribe_model === 'MOCK' ? 'Not a scientific result' : 'fsaverage5 output'}
        />
        <Card
          label="Stimulus"
          value={cards.stimulus}
          accent={cards.stimulus === 'MISSING' ? 'var(--color-ember)' : 'var(--color-vital)'}
          hint={cards.stimulus === 'MISSING' ? 'Awaiting licensed clip' : 'Validated'}
        />
        <Card
          label="Pipeline"
          value={<span className="text-base">{cards.pipeline}</span>}
          accent={pipelineAccent}
          hint={`${fmt.pct(cards.overall_progress)} of stages complete`}
        >
          <div className="mt-2.5">
            <Progress
              value={cards.overall_progress}
              state={cards.pipeline === 'RUNNING' ? 'RUNNING' : 'DONE'}
            />
          </div>
        </Card>
      </div>

      {/* Stage groups */}
      <div>
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-[1.05rem] font-semibold tracking-tight">Pipeline</h2>
            <p className="text-[0.79rem]" style={{ color: 'var(--fg-muted)' }}>
              A stage blocked on an external input never fails the run — only its
              dependants wait.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Legend />
            <button
              type="button"
              onClick={runTick}
              disabled={busy}
              className="rounded-lg px-3 py-1.5 text-[0.78rem] font-semibold transition-opacity disabled:opacity-50"
              style={{ background: 'var(--color-signal)', color: 'white' }}
            >
              {busy ? 'Running…' : 'Run tick now'}
            </button>
          </div>
        </div>

        {notice && (
          <div className="panel mb-3 px-3 py-2 text-[0.78rem]" style={{ color: 'var(--fg-muted)' }}>
            {notice}
          </div>
        )}

        <div className="grid gap-3 lg:grid-cols-2 xl:grid-cols-3">
          {dashboard.pipeline.groups.map((group) => (
            <div key={group.name} className="panel p-3.5">
              <h3
                className="mb-2.5 text-[0.68rem] font-semibold uppercase tracking-[0.1em]"
                style={{ color: 'var(--fg-faint)' }}
              >
                {group.name}
              </h3>
              <ul className="space-y-2">
                {group.stages.map((stage) => (
                  <StageRow key={stage.key} stage={stage} onRetry={retry} busy={busy} />
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function StageRow({
  stage,
  onRetry,
  busy,
}: {
  stage: Stage;
  onRetry: (key: string) => void;
  busy: boolean;
}) {
  const [open, setOpen] = useState(false);
  const color = STATE_COLOR[stage.state] ?? 'var(--fg-muted)';
  const retryable =
    stage.state === 'FAILED_RETRYABLE' ||
    stage.state === 'FAILED_FINAL' ||
    stage.state === 'WAITING_EXTERNAL';

  return (
    <li className="rounded-lg" style={{ background: 'var(--glass)' }}>
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="flex w-full items-start gap-2.5 px-2.5 py-2 text-left"
      >
        <span
          className={`mt-1 h-2 w-2 shrink-0 rounded-full ${stage.state === 'RUNNING' ? 'animate-pulse-soft' : ''}`}
          style={{ background: color, boxShadow: `0 0 10px ${color}` }}
          aria-hidden="true"
        />
        <span className="min-w-0 flex-1">
          <span className="flex items-center justify-between gap-2">
            <span className="truncate text-[0.83rem] font-medium">{stage.label}</span>
            {stage.attempts > 1 && (
              <span className="shrink-0 text-[0.65rem] numeric" style={{ color: 'var(--fg-faint)' }}>
                ×{stage.attempts}
              </span>
            )}
          </span>
          {stage.detail && (
            <span
              className={`mt-0.5 block text-[0.72rem] leading-snug ${open ? '' : 'line-clamp-1'}`}
              style={{ color: 'var(--fg-muted)' }}
            >
              {stage.detail}
            </span>
          )}
        </span>
      </button>

      {open && (
        <div className="space-y-2 border-t px-2.5 py-2">
          <div className="flex flex-wrap items-center gap-2">
            <Badge state={stage.state} />
            <span className="text-[0.68rem] numeric" style={{ color: 'var(--fg-faint)' }}>
              phase {stage.phase}
            </span>
          </div>
          {stage.description && (
            <p className="text-[0.72rem]" style={{ color: 'var(--fg-muted)' }}>
              {stage.description}
            </p>
          )}
          {stage.last_error && (
            <p className="text-[0.72rem] leading-snug" style={{ color: 'var(--color-alarm)' }}>
              {stage.last_error}
            </p>
          )}
          {stage.depends_on.length > 0 && (
            <p className="text-[0.68rem]" style={{ color: 'var(--fg-faint)' }}>
              depends on: {stage.depends_on.join(', ')}
            </p>
          )}
          {retryable && (
            <button
              type="button"
              onClick={() => onRetry(stage.key)}
              disabled={busy}
              className="rounded-md px-2 py-1 text-[0.72rem] font-semibold disabled:opacity-50"
              style={{
                background: 'color-mix(in oklab, var(--color-signal) 18%, transparent)',
                color: 'var(--color-signal)',
              }}
            >
              Re-check now
            </button>
          )}
        </div>
      )}
    </li>
  );
}

function Legend() {
  const items: [string, string][] = [
    ['DONE', 'done'],
    ['RUNNING', 'running'],
    ['WAITING_EXTERNAL', 'waiting'],
    ['FAILED_FINAL', 'failed'],
    ['BLOCKED', 'blocked'],
  ];
  return (
    <div className="hidden items-center gap-2.5 text-[0.66rem] md:flex" style={{ color: 'var(--fg-faint)' }}>
      {items.map(([state, label]) => (
        <span key={state} className="flex items-center gap-1">
          <span
            className="inline-block h-1.5 w-1.5 rounded-full"
            style={{ background: STATE_COLOR[state] }}
          />
          {label}
        </span>
      ))}
    </div>
  );
}
