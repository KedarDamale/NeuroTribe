/**
 * Subject Explorer.
 *
 * The synchronised view the specification asks for: one master timestamp links
 * the cortical map, the deviation timeline, the peak-moment list and (when a
 * licensed clip is present) the movie player.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, apiBase, fmt, type PeakWindow, type SubjectSummary } from '../../lib/api';
import CorticalViewer, { type ColorMap } from './CorticalViewer';
import { Badge, Empty, Section, Table, Td } from './ui';

interface SubjectDetail {
  external_id: string;
  site: string | null;
  age: number | null;
  sex: string | null;
  release: string | null;
  commercial_use_allowed: boolean | null;
  has_phenotype: boolean;
  diagnoses: {
    ordinal: number; label: string; certainty: string;
    category: string | null; is_adhd: boolean; is_no_diagnosis: boolean;
  }[];
  scans: {
    id: string; task: string | null; movie: string; movie_confidence: number | null;
    repetition_time: number | null; n_volumes: number | null; duration_sec: number | null;
    site: string | null; scanner: string | null; qc_status: string | null; mean_fd: number | null;
  }[];
  preprocessing: {
    status: string; engine: string; version: string | null; denoise_strategy: string | null;
    n_volumes: number | null; n_usable_frames: number | null;
    usable_frame_fraction: number | null; mean_fd: number | null;
    n_nonsteady_state: number | null; is_approximate: boolean; error: string | null;
  } | null;
  comparison: {
    id: string; valid: boolean; invalid_reason: string | null; movie: string;
    tr: number | null; global_agreement_r: number | null; global_mad: number | null;
    global_residual_variance: number | null; n_shared_timepoints: number | null;
    n_usable_timepoints: number | null; usable_frame_fraction: number | null;
    is_approximate: boolean; peak_windows: PeakWindow[];
    alignment_report: Record<string, unknown>;
    sanity_report: { valid: boolean; verdict: string; failures: string[]; warnings: string[] };
    top_deviation_rois: { roi_name: string; network: string | null; hemisphere: string | null; mad: number | null; agreement_r: number | null }[];
    top_deviation_networks: { network: string; mad: number | null; agreement_r: number | null }[];
    networks: { network: string; agreement_r: number | null; mad: number | null; n_vertices: number | null }[];
  } | null;
}

interface Timeline {
  movie: string;
  tr: number | null;
  peak_windows: PeakWindow[];
  timecourses: { time_sec: number[]; usable: boolean[]; global_deviation: (number | null)[] };
  rolling: { starts: number[]; ends: number[]; deviation: (number | null)[]; coverage: number[] } | null;
}

const MAPS: { key: string; label: string; colorMap: ColorMap; legend: string }[] = [
  { key: 'agreement', label: 'TRIBE agreement', colorMap: 'agreement', legend: 'Pearson r' },
  { key: 'deviation', label: 'Deviation', colorMap: 'deviation', legend: 'mean |residual|' },
];

export default function SubjectExplorer({ subjects }: { subjects: SubjectSummary[] }) {
  const withData = useMemo(() => subjects.filter((s) => s.comparison), [subjects]);
  const [selected, setSelected] = useState<string | null>(withData[0]?.external_id ?? null);
  const [detail, setDetail] = useState<SubjectDetail | null>(null);
  const [timeline, setTimeline] = useState<Timeline | null>(null);
  const [mapKind, setMapKind] = useState<string>('deviation');
  const [cursorSec, setCursorSec] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);
  const [stimulusKey, setStimulusKey] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    if (!selected) return;
    let cancelled = false;
    setError(null);
    setDetail(null);
    setTimeline(null);

    (async () => {
      try {
        const loaded = await api.get<SubjectDetail>(`/subjects/${selected}`);
        if (cancelled) return;
        setDetail(loaded);
        if (loaded.comparison?.valid) {
          try {
            const course = await api.get<Timeline>(`/subjects/${selected}/timeline`);
            if (!cancelled) setTimeline(course);
          } catch {
            /* the timeline is optional — the maps still work without it */
          }
        }
      } catch (cause) {
        if (!cancelled) setError((cause as Error).message);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [selected]);

  useEffect(() => {
    (async () => {
      try {
        const payload = await api.get<{ primary: string | null; stimuli: { key: string; validated: boolean }[] }>(
          '/stimulus',
        );
        setStimulusKey(payload.primary);
      } catch {
        setStimulusKey(null);
      }
    })();
  }, []);

  const seek = useCallback((seconds: number) => {
    setCursorSec(seconds);
    const video = videoRef.current;
    if (video && Number.isFinite(seconds)) {
      video.currentTime = Math.max(0, seconds);
    }
  }, []);

  if (!subjects.length) {
    return (
      <Empty
        title="No participants indexed yet"
        body="Index an HBN BIDS repository and the participants will appear here."
      />
    );
  }

  const active = MAPS.find((m) => m.key === mapKind) ?? MAPS[1];
  const valuesUrl =
    selected && detail?.comparison?.valid
      ? `/subjects/${selected}/map/${mapKind}?format=binary`
      : null;

  return (
    <div className="grid gap-5 xl:grid-cols-[19rem_minmax(0,1fr)]">
      {/* Roster */}
      <aside className="panel max-h-[36rem] overflow-y-auto p-2 xl:max-h-[calc(100dvh-11rem)] xl:sticky xl:top-24">
        <div className="px-2 py-1.5 text-[0.66rem] font-semibold uppercase tracking-[0.09em]" style={{ color: 'var(--fg-faint)' }}>
          {subjects.length} participants · {withData.length} analysed
        </div>
        <ul className="space-y-0.5">
          {subjects.map((subject) => {
            const isActive = subject.external_id === selected;
            const adhd = subject.diagnoses.some((d) => d.is_adhd && d.certainty === 'Confirmed');
            return (
              <li key={subject.external_id}>
                <button
                  type="button"
                  onClick={() => setSelected(subject.external_id)}
                  aria-current={isActive ? 'true' : undefined}
                  className="w-full rounded-lg px-2.5 py-2 text-left transition-colors"
                  style={
                    isActive
                      ? { background: 'color-mix(in oklab, var(--color-signal) 16%, transparent)' }
                      : undefined
                  }
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate text-[0.79rem] font-medium numeric">
                      {subject.external_id}
                    </span>
                    {adhd && (
                      <span
                        className="shrink-0 rounded px-1 py-px text-[0.58rem] font-bold"
                        style={{ background: 'color-mix(in oklab, var(--color-plasma) 20%, transparent)', color: 'var(--color-plasma)' }}
                      >
                        ADHD
                      </span>
                    )}
                  </div>
                  <div className="mt-0.5 flex items-center gap-2 text-[0.68rem]" style={{ color: 'var(--fg-muted)' }}>
                    <span>{subject.site ?? '—'}</span>
                    {subject.comparison ? (
                      <span className="numeric">r={fmt.num(subject.comparison.global_agreement_r, 2)}</span>
                    ) : (
                      <span style={{ color: 'var(--fg-faint)' }}>not analysed</span>
                    )}
                  </div>
                </button>
              </li>
            );
          })}
        </ul>
      </aside>

      {/* Detail */}
      <div className="min-w-0 space-y-5">
        {error && (
          <div className="panel px-4 py-3 text-[0.8rem]" style={{ color: 'var(--color-alarm)' }}>
            {error}
          </div>
        )}

        {detail && (
          <>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Fact label="Participant" value={detail.external_id} />
              <Fact
                label="Diagnosis"
                value={
                  detail.diagnoses.length
                    ? detail.diagnoses[0].label
                    : detail.has_phenotype
                      ? 'None recorded'
                      : 'Phenotype pending'
                }
                hint={detail.diagnoses[0]?.certainty}
              />
              <Fact
                label="Age / sex / site"
                value={`${detail.age ?? '—'} · ${detail.sex ?? '—'}`}
                hint={detail.site ?? undefined}
              />
              <Fact
                label="Usable fMRI"
                value={fmt.pct(detail.comparison?.usable_frame_fraction ?? detail.preprocessing?.usable_frame_fraction)}
                hint={
                  detail.preprocessing?.mean_fd != null
                    ? `mean FD ${fmt.num(detail.preprocessing.mean_fd, 3)} mm`
                    : undefined
                }
              />
            </div>

            {detail.comparison && !detail.comparison.valid && (
              <div
                className="panel px-4 py-3"
                style={{ borderColor: 'color-mix(in oklab, var(--color-alarm) 45%, transparent)' }}
              >
                <div className="mb-1 text-[0.78rem] font-bold" style={{ color: 'var(--color-alarm)' }}>
                  {detail.comparison.sanity_report?.verdict ?? 'ANALYSIS INVALID'}
                </div>
                <p className="text-[0.78rem]" style={{ color: 'var(--fg-muted)' }}>
                  {detail.comparison.invalid_reason}
                </p>
              </div>
            )}

            {detail.comparison?.is_approximate && (
              <div className="panel px-4 py-2.5 text-[0.78rem]" style={{ color: 'var(--color-ember)' }}>
                Approximate development projection — excluded from final analysis.
              </div>
            )}

            {!detail.comparison && (
              <Empty
                title="No TRIBE comparison for this participant yet"
                body="A comparison needs preprocessed fsaverage5 surfaces and a TRIBE prediction for the same stimulus."
              />
            )}

            {detail.comparison?.valid && (
              <>
                <Section
                  title="Cortical response"
                  description="Drag to orbit, scroll to zoom, hover to read a vertex."
                  actions={
                    <div className="glass flex gap-0.5 rounded-lg p-0.5">
                      {MAPS.map((option) => (
                        <button
                          key={option.key}
                          type="button"
                          onClick={() => setMapKind(option.key)}
                          aria-pressed={mapKind === option.key}
                          className="rounded-md px-2.5 py-1 text-[0.74rem] font-semibold transition-colors"
                          style={
                            mapKind === option.key
                              ? { background: 'var(--color-signal)', color: 'white' }
                              : { color: 'var(--fg-muted)' }
                          }
                        >
                          {option.label}
                        </button>
                      ))}
                    </div>
                  }
                >
                  <CorticalViewer
                    valuesUrl={valuesUrl}
                    colorMap={active.colorMap}
                    legendLabel={active.legend}
                    height={460}
                  />
                  <div className="mt-3 grid gap-3 sm:grid-cols-3">
                    <Fact
                      label="Global agreement"
                      value={fmt.num(detail.comparison.global_agreement_r, 3)}
                      hint="mean vertex-wise Pearson r"
                    />
                    <Fact
                      label="Global deviation"
                      value={fmt.num(detail.comparison.global_mad, 3)}
                      hint="mean absolute standardized residual"
                    />
                    <Fact
                      label="Timepoints"
                      value={`${fmt.int(detail.comparison.n_usable_timepoints)} / ${fmt.int(detail.comparison.n_shared_timepoints)}`}
                      hint="usable after censoring"
                    />
                  </div>
                </Section>

                {timeline && (
                  <Section
                    title="Deviation timeline"
                    description="Click anywhere to jump the whole view — including the movie — to that moment."
                  >
                    <DeviationTimeline
                      timeline={timeline}
                      cursorSec={cursorSec}
                      onSeek={seek}
                    />
                  </Section>
                )}

                <div className="grid gap-5 lg:grid-cols-2">
                  <Section
                    title="Peak deviation moments"
                    description="When did this brain diverge most from the normative prediction?"
                  >
                    <ol className="panel divide-y">
                      {detail.comparison.peak_windows.slice(0, 8).map((window) => (
                        <li key={window.rank}>
                          <button
                            type="button"
                            onClick={() => seek(window.start_sec)}
                            className="flex w-full items-center gap-3 px-3.5 py-2.5 text-left transition-colors hover:bg-[var(--glass)]"
                          >
                            <span
                              className="grid h-6 w-6 shrink-0 place-items-center rounded-md text-[0.68rem] font-bold numeric"
                              style={{ background: 'var(--glass)', color: 'var(--fg-muted)' }}
                            >
                              {window.rank}
                            </span>
                            <span className="flex-1">
                              <span className="block text-[0.82rem] font-semibold numeric">
                                {window.start_label}–{window.end_label}
                              </span>
                              <span className="text-[0.7rem]" style={{ color: 'var(--fg-faint)' }}>
                                coverage {fmt.pct(window.coverage)}
                              </span>
                            </span>
                            <span className="numeric text-[0.82rem] font-semibold" style={{ color: 'var(--color-ember)' }}>
                              {fmt.num(window.deviation, 3)}
                            </span>
                          </button>
                        </li>
                      ))}
                    </ol>
                  </Section>

                  <Section
                    title="Networks"
                    description="Aggregate scores. A single network value carries no medical meaning on its own."
                  >
                    <Table columns={['Network', 'Agreement r', 'Deviation', 'Vertices']} dense>
                      {detail.comparison.networks.map((row) => (
                        <tr key={row.network}>
                          <Td>{row.network}</Td>
                          <Td numeric>{fmt.num(row.agreement_r, 3)}</Td>
                          <Td numeric>{fmt.num(row.mad, 3)}</Td>
                          <Td numeric muted>{fmt.int(row.n_vertices)}</Td>
                        </tr>
                      ))}
                    </Table>
                  </Section>
                </div>

                {stimulusKey && (
                  <Section
                    title="Stimulus"
                    description="Synchronised with the timeline above via one shared master timestamp."
                  >
                    <div className="panel overflow-hidden">
                      <video
                        ref={videoRef}
                        controls
                        preload="metadata"
                        className="w-full"
                        style={{ maxHeight: 360, background: '#000' }}
                        onTimeUpdate={(event) => setCursorSec(event.currentTarget.currentTime)}
                      >
                        <source src={`${apiBase()}/stimulus/${stimulusKey}/media`} />
                        Your browser cannot play this clip.
                      </video>
                    </div>
                  </Section>
                )}

                <details className="panel p-4">
                  <summary className="cursor-pointer text-[0.86rem] font-semibold">
                    Alignment &amp; sanity report
                  </summary>
                  <div className="mt-3 space-y-2 text-[0.75rem]" style={{ color: 'var(--fg-muted)' }}>
                    {detail.comparison.sanity_report?.warnings?.map((warning, index) => (
                      <p key={index} style={{ color: 'var(--color-ember)' }}>
                        ⚠ {warning}
                      </p>
                    ))}
                    <pre
                      className="mt-2 max-h-72 overflow-auto rounded-lg p-3 text-[0.68rem] numeric"
                      style={{ background: 'var(--glass)' }}
                    >
                      {JSON.stringify(detail.comparison.alignment_report, null, 2)}
                    </pre>
                  </div>
                </details>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function Fact({ label, value, hint }: { label: string; value: React.ReactNode; hint?: string }) {
  return (
    <div className="panel p-3.5">
      <div className="text-[0.63rem] font-semibold uppercase tracking-[0.09em]" style={{ color: 'var(--fg-faint)' }}>
        {label}
      </div>
      <div className="mt-1 truncate text-[1.05rem] font-semibold numeric">{value}</div>
      {hint && (
        <div className="mt-0.5 truncate text-[0.7rem]" style={{ color: 'var(--fg-muted)' }}>
          {hint}
        </div>
      )}
    </div>
  );
}

/**
 * Sparkline-style deviation strip.
 *
 * Censored frames are drawn as gaps rather than interpolated, so the viewer can
 * never mistake removed data for a quiet period.
 */
function DeviationTimeline({
  timeline,
  cursorSec,
  onSeek,
}: {
  timeline: Timeline;
  cursorSec: number;
  onSeek: (seconds: number) => void;
}) {
  const { time_sec: times, global_deviation: deviation, usable } = timeline.timecourses;
  const width = 1000;
  const height = 90;

  const { path, gaps, tMin, tMax, vMax } = useMemo(() => {
    const finite = deviation.filter((v): v is number => v !== null && Number.isFinite(v));
    const max = finite.length ? Math.max(...finite) : 1;
    const low = times[0] ?? 0;
    const high = times[times.length - 1] ?? 1;
    const span = high - low || 1;

    const segments: string[] = [];
    const gapRects: { x: number; w: number }[] = [];
    let current = '';
    let gapStart: number | null = null;

    times.forEach((t, index) => {
      const x = ((t - low) / span) * width;
      const value = deviation[index];
      if (value === null || !Number.isFinite(value) || !usable[index]) {
        if (current) {
          segments.push(current);
          current = '';
        }
        if (gapStart === null) gapStart = x;
        return;
      }
      if (gapStart !== null) {
        gapRects.push({ x: gapStart, w: Math.max(1, x - gapStart) });
        gapStart = null;
      }
      const y = height - (value / (max || 1)) * (height - 8) - 4;
      current += `${current ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`;
    });
    if (current) segments.push(current);
    if (gapStart !== null) gapRects.push({ x: gapStart, w: width - gapStart });

    return { path: segments.join(' '), gaps: gapRects, tMin: low, tMax: high, vMax: max };
  }, [times, deviation, usable]);

  const cursorX = ((cursorSec - tMin) / (tMax - tMin || 1)) * width;

  return (
    <div className="panel p-3">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full cursor-crosshair"
        style={{ height: 110 }}
        role="img"
        aria-label="Global deviation over the stimulus"
        onClick={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          const ratio = (event.clientX - rect.left) / rect.width;
          onSeek(tMin + ratio * (tMax - tMin));
        }}
      >
        <defs>
          <linearGradient id="dev-stroke" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--color-ember)" />
            <stop offset="100%" stopColor="var(--color-plasma)" />
          </linearGradient>
        </defs>

        {gaps.map((gap, index) => (
          <rect
            key={index}
            x={gap.x}
            y={0}
            width={gap.w}
            height={height}
            fill="var(--fg-faint)"
            opacity={0.12}
          />
        ))}

        <path d={path} fill="none" stroke="url(#dev-stroke)" strokeWidth={1.6} strokeLinejoin="round" />

        {timeline.peak_windows.slice(0, 5).map((window) => {
          const x = ((window.start_sec - tMin) / (tMax - tMin || 1)) * width;
          const w = ((window.end_sec - window.start_sec) / (tMax - tMin || 1)) * width;
          return (
            <rect
              key={window.rank}
              x={x}
              y={0}
              width={Math.max(2, w)}
              height={height}
              fill="var(--color-ember)"
              opacity={0.14}
            />
          );
        })}

        {Number.isFinite(cursorX) && cursorX >= 0 && cursorX <= width && (
          <line
            x1={cursorX}
            x2={cursorX}
            y1={0}
            y2={height}
            stroke="var(--color-signal)"
            strokeWidth={1.5}
          />
        )}
      </svg>

      <div className="mt-1 flex justify-between text-[0.66rem] numeric" style={{ color: 'var(--fg-faint)' }}>
        <span>{fmt.timecode(tMin)}</span>
        <span>
          peak {fmt.num(vMax, 3)} · cursor {fmt.timecode(cursorSec)} · shaded = censored
        </span>
        <span>{fmt.timecode(tMax)}</span>
      </div>
    </div>
  );
}
