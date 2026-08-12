import { useState } from 'react';
import { api } from '../../lib/api';
import { Empty, usePolling } from './ui';

interface Entry {
  ts: string;
  level: 'INFO' | 'WARNING' | 'ERROR';
  logger: string;
  message: string;
  context: Record<string, unknown>;
}

const LEVEL_COLOR: Record<string, string> = {
  INFO: 'var(--color-signal)',
  WARNING: 'var(--color-ember)',
  ERROR: 'var(--color-alarm)',
};

export default function LogStream() {
  const [level, setLevel] = useState<string>('');
  const { data } = usePolling<{ entries: Entry[]; note?: string }>(
    () => api.get(`/logs?limit=150${level ? `&level=${level}` : ''}`),
    5000,
  );

  const entries = data?.entries ?? [];

  return (
    <div className="space-y-2.5">
      <div className="flex gap-1.5">
        {['', 'INFO', 'WARNING', 'ERROR'].map((option) => (
          <button
            key={option || 'all'}
            type="button"
            onClick={() => setLevel(option)}
            aria-pressed={level === option}
            className="rounded-lg px-2.5 py-1 text-[0.73rem] font-semibold transition-colors"
            style={
              level === option
                ? { background: LEVEL_COLOR[option] ?? 'var(--color-signal)', color: 'white' }
                : { background: 'var(--glass)', color: 'var(--fg-muted)' }
            }
          >
            {option || 'All'}
          </button>
        ))}
      </div>

      {entries.length === 0 ? (
        <Empty title="No log entries yet" body={data?.note} />
      ) : (
        <div className="panel max-h-[26rem] overflow-y-auto p-1">
          <ul className="divide-y">
            {entries.map((entry, index) => (
              <li key={index} className="flex gap-2.5 px-2.5 py-1.5 text-[0.73rem]">
                <span className="shrink-0 numeric" style={{ color: 'var(--fg-faint)' }}>
                  {entry.ts?.slice(11, 19)}
                </span>
                <span
                  className="w-14 shrink-0 font-bold"
                  style={{ color: LEVEL_COLOR[entry.level] ?? 'var(--fg-muted)' }}
                >
                  {entry.level}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block">{entry.message}</span>
                  <span className="text-[0.66rem] numeric" style={{ color: 'var(--fg-faint)' }}>
                    {entry.logger}
                  </span>
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
