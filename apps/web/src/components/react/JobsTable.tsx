import { api, fmt } from '../../lib/api';
import { Badge, Empty, Progress, Table, Td, usePolling } from './ui';

interface Job {
  id: string;
  name: string;
  kind: string;
  stage_key: string | null;
  subject_external_id: string | null;
  state: string;
  progress: number;
  message: string | null;
  started_at: string | null;
  elapsed_sec: number | null;
  eta_sec: number | null;
  retry_count: number;
  cache_hit: boolean;
  cpu_percent: number | null;
  mem_mb: number | null;
  gpu_name: string | null;
  error_message: string | null;
  has_log: boolean;
}

interface Payload {
  jobs: Job[];
  by_state: Record<string, number>;
  active: number;
}

export default function JobsTable({ initial }: { initial: Payload }) {
  const { data } = usePolling<Payload>(() => api.get<Payload>('/jobs?limit=120'), 4000);
  const payload = data ?? initial;

  if (!payload.jobs.length) {
    return (
      <Empty
        title="No jobs yet"
        body="Jobs appear as soon as the Autopilot starts advancing the pipeline."
      />
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        {Object.entries(payload.by_state).map(([state, count]) => (
          <Badge key={state} state={state}>
            {state.toLowerCase()} · {count}
          </Badge>
        ))}
      </div>

      <Table columns={['Job', 'Kind', 'State', 'Progress', 'Elapsed', 'Resources', 'Message']}>
        {payload.jobs.map((job) => (
          <tr key={job.id}>
            <Td title={job.name}>
              <span className="block max-w-[18rem] truncate font-medium">{job.name}</span>
              {job.subject_external_id && (
                <span className="text-[0.68rem] numeric" style={{ color: 'var(--fg-faint)' }}>
                  {job.subject_external_id}
                </span>
              )}
            </Td>
            <Td muted>{job.kind}</Td>
            <Td>
              <Badge state={job.state}>
                {job.cache_hit ? 'cached' : job.state.toLowerCase()}
              </Badge>
            </Td>
            <Td>
              <div className="w-24">
                <Progress value={job.progress} state={job.state} />
              </div>
            </Td>
            <Td numeric muted>{fmt.duration(job.elapsed_sec)}</Td>
            <Td numeric muted>
              {job.cpu_percent != null ? `${job.cpu_percent.toFixed(0)}% cpu` : '—'}
              {job.mem_mb != null ? ` · ${(job.mem_mb / 1024).toFixed(1)} GB` : ''}
            </Td>
            <Td muted title={job.error_message ?? job.message ?? ''}>
              <span
                className="block max-w-[22rem] truncate"
                style={job.error_message ? { color: 'var(--color-alarm)' } : undefined}
              >
                {job.error_message ?? job.message ?? '—'}
              </span>
            </Td>
          </tr>
        ))}
      </Table>
    </div>
  );
}
