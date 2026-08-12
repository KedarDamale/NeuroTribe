/**
 * Blockers panel — the most important surface in the application.
 *
 * The three legitimate external gates (HBN phenotype access, the licensed
 * stimulus, the FreeSurfer license) must be explained honestly rather than
 * hidden or faked. Each entry says what is required, why, and where to get it.
 */

import { api, type Blocker } from '../../lib/api';
import { Badge, Empty, usePolling } from './ui';

const SEVERITY_ORDER = ['EXTERNAL', 'ACTIONABLE', 'INFO'] as const;

const SEVERITY_COPY: Record<string, { title: string; body: string }> = {
  EXTERNAL: {
    title: 'External blockers',
    body: 'These require human or institutional action. Software cannot resolve them, and NeuroTRIBE will not fabricate a way around them.',
  },
  ACTIONABLE: {
    title: 'Actionable',
    body: 'The system can clear these once the local environment is fixed.',
  },
  INFO: {
    title: 'Advisory',
    body: 'Not blocking, but they affect how results should be interpreted.',
  },
};

export default function BlockersPanel({ initial }: { initial: Blocker[] }) {
  const { data } = usePolling<{ external: Blocker[]; actionable: Blocker[]; info: Blocker[] }>(
    () => api.get('/blockers'),
    8000,
  );

  const grouped = data ?? {
    external: initial.filter((b) => b.severity === 'EXTERNAL'),
    actionable: initial.filter((b) => b.severity === 'ACTIONABLE'),
    info: initial.filter((b) => b.severity === 'INFO'),
  };

  const total = grouped.external.length + grouped.actionable.length + grouped.info.length;

  if (total === 0) {
    return (
      <Empty
        title="No active blockers"
        body="Every input the pipeline needs is present. The Autopilot will keep advancing on its own."
      />
    );
  }

  return (
    <div className="space-y-5">
      {SEVERITY_ORDER.map((severity) => {
        const key = severity.toLowerCase() as 'external' | 'actionable' | 'info';
        const items = grouped[key];
        if (!items.length) return null;
        const copy = SEVERITY_COPY[severity];
        return (
          <div key={severity}>
            <div className="mb-2">
              <h3 className="flex items-center gap-2 text-[0.95rem] font-semibold tracking-tight">
                {copy.title}
                <Badge state={severity} dot={false}>
                  {items.length}
                </Badge>
              </h3>
              <p className="mt-0.5 max-w-2xl text-[0.78rem]" style={{ color: 'var(--fg-muted)' }}>
                {copy.body}
              </p>
            </div>
            <div className="grid gap-3 lg:grid-cols-2">
              {items.map((blocker) => (
                <BlockerCard key={blocker.id} blocker={blocker} />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function BlockerCard({ blocker }: { blocker: Blocker }) {
  const accent =
    blocker.severity === 'EXTERNAL'
      ? 'var(--color-ember)'
      : blocker.severity === 'ACTIONABLE'
        ? 'var(--color-signal)'
        : 'var(--fg-muted)';

  const path = (blocker.context?.incoming_dir ?? blocker.context?.path) as string | undefined;

  return (
    <article className="panel relative overflow-hidden p-4">
      <div className="absolute inset-y-0 left-0 w-0.5" style={{ background: accent }} aria-hidden="true" />
      <header className="mb-2 flex flex-wrap items-start justify-between gap-2 pl-2">
        <h4 className="text-[0.92rem] font-semibold tracking-tight">{blocker.title}</h4>
        <Badge state={blocker.severity} dot={false}>
          {blocker.kind.replaceAll('_', ' ').toLowerCase()}
        </Badge>
      </header>

      <p className="pl-2 text-[0.79rem] leading-relaxed" style={{ color: 'var(--fg-muted)' }}>
        {blocker.description}
      </p>

      {blocker.required_action && (
        <div
          className="mt-3 ml-2 rounded-lg px-3 py-2 text-[0.76rem] leading-relaxed"
          style={{ background: 'var(--glass)', border: '1px solid var(--glass-line)' }}
        >
          <div
            className="mb-1 text-[0.62rem] font-semibold uppercase tracking-[0.09em]"
            style={{ color: accent }}
          >
            Required action
          </div>
          {blocker.required_action}
        </div>
      )}

      {path && (
        <div
          className="mt-2 ml-2 overflow-x-auto rounded-md px-2.5 py-1.5 text-[0.7rem] numeric"
          style={{ background: 'var(--glass)', color: 'var(--fg-muted)' }}
        >
          {path}
        </div>
      )}

      <footer className="mt-3 flex flex-wrap items-center gap-3 pl-2">
        {blocker.reference_url && (
          <a
            href={blocker.reference_url}
            target="_blank"
            rel="noreferrer noopener"
            className="text-[0.75rem] font-semibold underline underline-offset-2"
            style={{ color: 'var(--color-signal)' }}
          >
            Reference ↗
          </a>
        )}
        {blocker.blocks_stages.length > 0 && (
          <span className="text-[0.7rem]" style={{ color: 'var(--fg-faint)' }}>
            blocks {blocker.blocks_stages.length} downstream stage
            {blocker.blocks_stages.length === 1 ? '' : 's'}
          </span>
        )}
      </footer>
    </article>
  );
}
