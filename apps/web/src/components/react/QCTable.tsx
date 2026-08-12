import { useMemo, useState } from 'react';
import { fmt } from '../../lib/api';
import { Badge, Card, Empty, Table, Td } from './ui';

interface QCRow {
  subject_external_id: string;
  site: string | null;
  group: string | null;
  preprocessing: string;
  anatomical: string;
  bold: string;
  motion: string;
  mriqc: string;
  alignment: string;
  usable_frame_fraction: number | null;
  mean_fd: number | null;
  is_approximate: boolean;
  overall: string;
  notes: string[];
}

interface Payload {
  rows: QCRow[];
  summary: {
    n_rows: number;
    by_status: Record<string, number>;
    n_approximate: number;
    median_usable_frame_fraction: number | null;
    median_mean_fd: number | null;
  };
  sites: string[];
  groups: { key: string; label: string }[];
  policy: Record<string, number | null>;
}

export default function QCTable({ initial }: { initial: Payload }) {
  const [onlyFailures, setOnlyFailures] = useState(false);
  const [site, setSite] = useState('');
  const [group, setGroup] = useState('');

  const rows = useMemo(
    () =>
      initial.rows.filter((row) => {
        if (onlyFailures && row.overall !== 'FAIL' && row.overall !== 'WARNING') return false;
        if (site && row.site !== site) return false;
        if (group && row.group !== group) return false;
        return true;
      }),
    [initial.rows, onlyFailures, site, group],
  );

  if (!initial.rows.length) {
    return (
      <Empty
        title="No participants with imaging yet"
        body="QC rows appear once an HBN BIDS repository has been indexed."
      />
    );
  }

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Card label="Participants" value={fmt.int(initial.summary.n_rows)} />
        <Card
          label="Passing"
          value={fmt.int(initial.summary.by_status.PASS ?? 0)}
          accent="var(--color-vital)"
        />
        <Card
          label="Median usable frames"
          value={fmt.pct(initial.summary.median_usable_frame_fraction)}
          hint={`policy ≥ ${fmt.pct(initial.policy.min_usable_frame_fraction as number)}`}
        />
        <Card
          label="Median mean FD"
          value={fmt.num(initial.summary.median_mean_fd, 3)}
          hint={`censor > ${initial.policy.fd_threshold_mm} mm`}
        />
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => setOnlyFailures((value) => !value)}
          aria-pressed={onlyFailures}
          className="rounded-lg px-3 py-1.5 text-[0.76rem] font-semibold"
          style={
            onlyFailures
              ? { background: 'var(--color-alarm)', color: 'white' }
              : { background: 'var(--glass)', color: 'var(--fg-muted)' }
          }
        >
          Only failures &amp; warnings
        </button>
        <select
          value={site}
          onChange={(event) => setSite(event.target.value)}
          aria-label="Filter by site"
          className="rounded-lg px-2.5 py-1.5 text-[0.76rem]"
          style={{ background: 'var(--glass)', color: 'var(--fg)', border: '1px solid var(--line)' }}
        >
          <option value="">All sites</option>
          {initial.sites.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
        <select
          value={group}
          onChange={(event) => setGroup(event.target.value)}
          aria-label="Filter by cohort group"
          className="rounded-lg px-2.5 py-1.5 text-[0.76rem]"
          style={{ background: 'var(--glass)', color: 'var(--fg)', border: '1px solid var(--line)' }}
        >
          <option value="">All groups</option>
          {initial.groups.map((option) => (
            <option key={option.key} value={option.key}>
              {option.label}
            </option>
          ))}
        </select>
        <span className="text-[0.74rem]" style={{ color: 'var(--fg-muted)' }}>
          {rows.length} shown
        </span>
      </div>

      <Table
        columns={[
          'Participant', 'Overall', 'Preprocessing', 'T1w', 'BOLD', 'Motion',
          'MRIQC', 'Alignment', 'Usable', 'Mean FD', 'Notes',
        ]}
      >
        {rows.map((row) => (
          <tr key={row.subject_external_id}>
            <Td numeric>
              {row.subject_external_id}
              {row.is_approximate && (
                <span className="ml-1.5 text-[0.62rem]" style={{ color: 'var(--color-ember)' }}>
                  approx
                </span>
              )}
            </Td>
            <Td><Badge state={row.overall} /></Td>
            <Td><Badge state={row.preprocessing} dot={false} /></Td>
            <Td><Badge state={row.anatomical} dot={false} /></Td>
            <Td><Badge state={row.bold} dot={false} /></Td>
            <Td><Badge state={row.motion} dot={false} /></Td>
            <Td><Badge state={row.mriqc} dot={false} /></Td>
            <Td><Badge state={row.alignment} dot={false} /></Td>
            <Td numeric>{fmt.pct(row.usable_frame_fraction)}</Td>
            <Td numeric muted>{fmt.num(row.mean_fd, 3)}</Td>
            <Td muted title={row.notes.join(' · ')}>
              <span className="block max-w-[20rem] truncate">{row.notes[0] ?? '—'}</span>
            </Td>
          </tr>
        ))}
      </Table>
    </div>
  );
}
