/**
 * Group analysis view.
 *
 * Reporting rules enforced here:
 *   - effect size, confidence interval and sample sizes are always shown; a
 *     significance marker never stands alone;
 *   - PRIMARY and EXPLORATORY results are visually distinct and never mixed;
 *   - the cortical effect map paints EFFECT SIZE, not p-values.
 */

import { useEffect, useMemo, useState } from 'react';
import { api, fmt, type GroupResultRow } from '../../lib/api';
import CorticalViewer from './CorticalViewer';
import { Badge, Empty, Section, Table, Td } from './ui';

interface RunMeta {
  id: string;
  name: string;
  tier: string;
  status: string;
  n_case: number;
  n_control: number;
  model_formula: string;
  correction: string;
  alpha: number;
  sanity_passed: boolean;
  sanity_report: { failures: string[]; warnings: string[] };
  summary: Record<string, unknown>;
  provenance: Record<string, unknown>;
  case_label: string;
  control_label: string;
}

interface Payload {
  available: boolean;
  reason?: string;
  run?: RunMeta;
  results?: GroupResultRow[];
  n_significant?: number;
  note?: string;
}

const METRICS = [
  { key: 'mad', label: 'Deviation (MAD)' },
  { key: 'agreement_r', label: 'TRIBE agreement' },
  { key: 'residual_variance', label: 'Residual variance' },
];

export default function GroupResults({ initial }: { initial: Payload }) {
  const [tier, setTier] = useState<'PRIMARY' | 'EXPLORATORY'>('PRIMARY');
  const [metric, setMetric] = useState('mad');
  const [unitType, setUnitType] = useState<'network' | 'roi' | 'global'>('network');
  const [payload, setPayload] = useState<Payload>(initial);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    (async () => {
      try {
        const result = await api.get<Payload>(
          `/groups/results?tier=${tier}&metric=${metric}`,
        );
        if (!cancelled) setPayload(result);
      } catch (cause) {
        if (!cancelled) {
          setPayload({ available: false, reason: (cause as Error).message });
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [tier, metric]);

  const rows = useMemo(
    () => (payload.results ?? []).filter((row) => row.unit_type === unitType),
    [payload.results, unitType],
  );

  if (!payload.available) {
    return (
      <Empty
        title="No group analysis yet"
        body={
          payload.reason ??
          'A group contrast needs a built cohort and at least one valid subject comparison per group.'
        }
      />
    );
  }

  const run = payload.run!;
  const alpha = run.alpha ?? 0.05;

  return (
    <div className="space-y-5">
      {/* Contrast header */}
      <div className="panel p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-3">
            <div>
              <div className="text-[0.63rem] font-semibold uppercase tracking-[0.09em]" style={{ color: 'var(--fg-faint)' }}>
                Case group
              </div>
              <div className="text-[0.95rem] font-semibold">
                {run.case_label}{' '}
                <span className="numeric" style={{ color: 'var(--fg-muted)' }}>
                  n={run.n_case}
                </span>
              </div>
            </div>
            <div className="px-1 text-lg" style={{ color: 'var(--fg-faint)' }} aria-hidden="true">
              vs
            </div>
            <div>
              <div className="text-[0.63rem] font-semibold uppercase tracking-[0.09em]" style={{ color: 'var(--fg-faint)' }}>
                Comparison group
              </div>
              <div className="text-[0.95rem] font-semibold">
                {run.control_label}{' '}
                <span className="numeric" style={{ color: 'var(--fg-muted)' }}>
                  n={run.n_control}
                </span>
              </div>
            </div>
          </div>

          <div className="glass flex gap-0.5 rounded-lg p-0.5">
            {(['PRIMARY', 'EXPLORATORY'] as const).map((option) => (
              <button
                key={option}
                type="button"
                onClick={() => setTier(option)}
                aria-pressed={tier === option}
                className="rounded-md px-3 py-1.5 text-[0.74rem] font-bold tracking-wide transition-colors"
                style={
                  tier === option
                    ? {
                        background: option === 'PRIMARY' ? 'var(--color-signal)' : 'var(--color-plasma)',
                        color: 'white',
                      }
                    : { color: 'var(--fg-muted)' }
                }
              >
                {option}
              </button>
            ))}
          </div>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-1.5 text-[0.74rem]" style={{ color: 'var(--fg-muted)' }}>
          <span className="numeric">model: {run.model_formula}</span>
          <span>correction: {run.correction}</span>
          <span className="numeric">α = {alpha}</span>
          <Badge state={run.sanity_passed ? 'DONE' : 'FAILED_FINAL'} dot={false}>
            {run.sanity_passed ? 'sanity passed' : 'ANALYSIS INVALID'}
          </Badge>
        </div>

        {!run.sanity_passed && run.sanity_report?.failures?.length > 0 && (
          <ul className="mt-2 space-y-1 text-[0.75rem]" style={{ color: 'var(--color-alarm)' }}>
            {run.sanity_report.failures.map((failure, index) => (
              <li key={index}>• {failure}</li>
            ))}
          </ul>
        )}
        {run.sanity_report?.warnings?.length > 0 && (
          <ul className="mt-2 space-y-1 text-[0.73rem]" style={{ color: 'var(--color-ember)' }}>
            {run.sanity_report.warnings.slice(0, 4).map((warning, index) => (
              <li key={index}>⚠ {warning}</li>
            ))}
          </ul>
        )}

        {tier === 'EXPLORATORY' && (
          <p className="mt-2 text-[0.73rem]" style={{ color: 'var(--color-plasma)' }}>
            Exploratory: hypothesis-generating only. It does not modify the primary result.
          </p>
        )}
      </div>

      {/* Effect map */}
      <Section
        title="Cortical effect map"
        description="ADHD minus comparison, expressed as Cohen's d — not a p-value map."
      >
        <CorticalViewer
          valuesUrl={`/groups/effect-map?tier=${tier}&metric=${metric}&format=binary`}
          colorMap="diverging"
          legendLabel="Cohen's d"
          height={440}
        />
      </Section>

      {/* Results table */}
      <Section
        title="Results"
        description="Effect sizes with 95% confidence intervals, FDR-adjusted q-values and per-group sample sizes."
        actions={
          <div className="flex flex-wrap gap-1.5">
            <div className="glass flex gap-0.5 rounded-lg p-0.5">
              {METRICS.map((option) => (
                <button
                  key={option.key}
                  type="button"
                  onClick={() => setMetric(option.key)}
                  aria-pressed={metric === option.key}
                  className="rounded-md px-2.5 py-1 text-[0.72rem] font-semibold transition-colors"
                  style={
                    metric === option.key
                      ? { background: 'var(--color-signal)', color: 'white' }
                      : { color: 'var(--fg-muted)' }
                  }
                >
                  {option.label}
                </button>
              ))}
            </div>
            <div className="glass flex gap-0.5 rounded-lg p-0.5">
              {(['network', 'roi', 'global'] as const).map((option) => (
                <button
                  key={option}
                  type="button"
                  onClick={() => setUnitType(option)}
                  aria-pressed={unitType === option}
                  className="rounded-md px-2.5 py-1 text-[0.72rem] font-semibold capitalize transition-colors"
                  style={
                    unitType === option
                      ? { background: 'var(--color-signal)', color: 'white' }
                      : { color: 'var(--fg-muted)' }
                  }
                >
                  {option}
                </button>
              ))}
            </div>
          </div>
        }
      >
        {loading && (
          <div className="mb-2 text-[0.75rem]" style={{ color: 'var(--fg-muted)' }}>
            Loading…
          </div>
        )}
        {rows.length === 0 ? (
          <Empty title="No results for this selection" />
        ) : (
          <Table
            columns={[
              unitType === 'roi' ? 'ROI' : 'Unit',
              'Network',
              'ADHD',
              'Comparison',
              "Cohen's d",
              '95% CI',
              'p',
              'q (FDR)',
              'n',
            ]}
          >
            {rows.map((row) => {
              const significant = row.q_value !== null && row.q_value < alpha;
              return (
                <tr
                  key={`${row.unit_type}-${row.unit_name}`}
                  style={
                    significant
                      ? { background: 'color-mix(in oklab, var(--color-signal) 7%, transparent)' }
                      : undefined
                  }
                >
                  <Td title={row.unit_name}>
                    <span className="flex items-center gap-1.5">
                      {significant && (
                        <span
                          className="inline-block h-1.5 w-1.5 rounded-full"
                          style={{ background: 'var(--color-signal)' }}
                          aria-label="survives FDR correction"
                        />
                      )}
                      <span className="max-w-[16rem] truncate">{row.unit_name}</span>
                    </span>
                  </Td>
                  <Td muted>{row.network ?? '—'}</Td>
                  <Td numeric>{fmt.num(row.mean_case)}</Td>
                  <Td numeric>{fmt.num(row.mean_control)}</Td>
                  <Td numeric>
                    <span
                      style={{
                        color:
                          row.effect_size === null
                            ? undefined
                            : row.effect_size > 0
                              ? 'var(--color-plasma)'
                              : 'var(--color-signal)',
                      }}
                    >
                      {fmt.num(row.effect_size, 2)}
                    </span>
                  </Td>
                  <Td numeric muted>
                    {row.ci_low === null || row.ci_high === null
                      ? '—'
                      : `[${fmt.num(row.ci_low, 2)}, ${fmt.num(row.ci_high, 2)}]`}
                  </Td>
                  <Td numeric muted>{fmt.num(row.p_value, 4)}</Td>
                  <Td numeric>
                    <strong style={{ color: significant ? 'var(--color-signal)' : undefined }}>
                      {fmt.num(row.q_value, 4)}
                    </strong>
                  </Td>
                  <Td numeric muted>
                    {row.n_case}/{row.n_control}
                  </Td>
                </tr>
              );
            })}
          </Table>
        )}

        <p className="mt-2.5 text-[0.73rem]" style={{ color: 'var(--fg-faint)' }}>
          {payload.note}
        </p>
      </Section>

      <details className="panel p-4">
        <summary className="cursor-pointer text-[0.86rem] font-semibold">
          Reproducibility manifest
        </summary>
        <pre
          className="mt-3 max-h-80 overflow-auto rounded-lg p-3 text-[0.68rem] numeric"
          style={{ background: 'var(--glass)' }}
        >
          {JSON.stringify(run.provenance, null, 2)}
        </pre>
      </details>
    </div>
  );
}
