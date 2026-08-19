import { useEffect, useState } from 'react';
import CorticalViewer from './CorticalViewer';
import { api, apiBase, publicApiBase } from '../../lib/api';

type Bold = { id: string; label: string; n_volumes: number; tr_sec: number };
type Image = { id: string; label: string };
type Catalog = { bold: Bold[]; images: Image[] };
type Run = { id?: string; status: string; progress: number; error?: string; summary?: { mean_deviation: number } };

export default function ExplorationStudio() {
  const [catalog, setCatalog] = useState<Catalog>({ bold: [], images: [] });
  const [boldId, setBoldId] = useState('');
  const [imageId, setImageId] = useState('');
  const [run, setRun] = useState<Run | null>(null);
  const [uploading, setUploading] = useState(false);
  const [predictionFile, setPredictionFile] = useState<File | null>(null);
  const [importingPrediction, setImportingPrediction] = useState(false);
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [catalogError, setCatalogError] = useState('');

  const refresh = async () => {
    setCatalogLoading(true);
    setCatalogError('');
    try {
      const next = await api.get<Catalog>('/exploratory/catalog');
      setCatalog(next);
      setBoldId((value) => value || next.bold[0]?.id || '');
      setImageId((value) => value || next.images[0]?.id || '');
    } catch (error) {
      setCatalogError(error instanceof Error ? error.message : 'Could not load the local data catalog.');
    } finally {
      setCatalogLoading(false);
    }
  };

  useEffect(() => { void refresh(); }, []);

  useEffect(() => {
    if (!run?.id || ['DONE', 'FAILED'].includes(run.status)) return;
    const timer = window.setInterval(() => {
      api.get<Run>(`/exploratory/runs/${run.id}`)
        .then((next) => setRun((current) => ({ ...current, ...next, id: next.id || current?.id })))
        .catch(() => undefined);
    }, 2500);
    return () => window.clearInterval(timer);
  }, [run?.id, run?.status]);

  const upload = async (file?: File) => {
    if (!file) return;
    setUploading(true);
    try {
      const body = new FormData();
      body.append('file', file);
      const response = await fetch(`${apiBase()}/exploratory/images`, { method: 'POST', body });
      if (!response.ok) throw new Error(await response.text());
      const image = await response.json() as Image;
      await refresh();
      setImageId(image.id);
    } finally {
      setUploading(false);
    }
  };

  const importPrediction = async () => {
    if (!predictionFile) return;
    setImportingPrediction(true);
    try {
      const body = new FormData();
      body.append('bold_id', boldId);
      body.append('image_id', imageId);
      body.append('prediction', predictionFile);
      const response = await fetch(`${apiBase()}/exploratory/runs/imported`, { method: 'POST', body });
      if (!response.ok) throw new Error(await response.text());
      setRun(await response.json() as Run);
    } catch (error) {
      setRun({ status: 'FAILED', progress: 1, error: error instanceof Error ? error.message : 'Could not import the TRIBE prediction.' });
    } finally {
      setImportingPrediction(false);
    }
  };
  const done = Boolean(run?.id && run.status === 'DONE');
  const map = (kind: 'observed' | 'predicted' | 'deviation') => done ? `/exploratory/runs/${run!.id}/maps/${kind}` : null;
  const selected = catalog.bold.find((item) => item.id === boldId);
  const disabled = catalogLoading || Boolean(catalogError);

  return <div className="space-y-6">
    <section className="panel p-5 lg:p-6">
      <p className="text-xs font-semibold uppercase tracking-[.12em]" style={{ color: 'var(--color-signal)' }}>Exploratory single-subject analysis</p>
      <div className="mt-2 grid gap-6 xl:grid-cols-[1.1fr_1fr]">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight">Image to TRIBE to cortical deviation</h2>
          <p className="mt-2 text-sm leading-relaxed" style={{ color: 'var(--fg-muted)' }}>Select local BIDS fMRI and an image. Run the supplied script in a free interactive GPU notebook, then upload its TRIBE prediction to calculate the three cortical maps locally.</p>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <label className="field"><span>fMRI data</span><select value={boldId} disabled={disabled} onChange={(event) => setBoldId(event.target.value)}><option value="">{catalogLoading ? 'Loading local fMRI...' : catalogError ? 'Catalog unavailable' : 'Choose fMRI data'}</option>{catalog.bold.map((item) => <option value={item.id} key={item.id}>{item.label}</option>)}</select></label>
          <label className="field"><span>Stimulus image</span><select value={imageId} disabled={disabled} onChange={(event) => setImageId(event.target.value)}><option value="">{catalogLoading ? 'Loading images...' : catalogError ? 'Catalog unavailable' : 'Choose stimulus image'}</option>{catalog.images.map((item) => <option value={item.id} key={item.id}>{item.label}</option>)}</select></label>
          <label className="upload-field sm:col-span-2"><span>{uploading ? 'Adding image...' : 'Upload image'}</span><input type="file" accept="image/png,image/jpeg,image/webp,image/bmp" onChange={(event) => void upload(event.target.files?.[0])} /></label>
          <label className="upload-field sm:col-span-2"><span>TRIBE prediction from free GPU notebook (.npy)</span><input type="file" accept=".npy,application/octet-stream" onChange={(event) => setPredictionFile(event.target.files?.[0] || null)} />{predictionFile && <small>{predictionFile.name}</small>}</label>
        </div>
      </div>
      {catalogError && <p className="mt-3 text-sm" style={{ color: 'var(--color-alarm)' }}>{catalogError} <button className="underline" onClick={() => void refresh()}>Retry</button></p>}
      <div className="mt-5 flex flex-wrap items-center gap-3 border-t pt-4" style={{ borderColor: 'var(--border)' }}>
        <a className="primary-button" href={`${publicApiBase()}/exploratory/free-gpu-script`}>Download free GPU script</a>
        <button className="primary-button" disabled={!boldId || !imageId || !predictionFile || importingPrediction || (!!run && !['DONE', 'FAILED'].includes(run.status))} onClick={() => void importPrediction()}>{importingPrediction ? 'Importing prediction...' : 'Analyze TRIBE prediction'}</button>
        {selected && <span className="text-xs" style={{ color: 'var(--fg-muted)' }}>{selected.n_volumes} volumes | TR {selected.tr_sec}s | 4s visual TRIBE clip | free mode</span>}
        {run && <span className="status-pill">{run.status.replace('_', ' ')} | {Math.round(run.progress * 100)}%</span>}
      </div>
      {run?.error && <p className="mt-3 text-sm" style={{ color: 'var(--color-alarm)' }}>{run.error}</p>}
    </section>
    <section className="grid gap-5 xl:grid-cols-3">
      <Map title="Observed fMRI" note="Volume projection | display only" values={map('observed')} color="diverging" label="observed signal" />
      <Map title="TRIBE prediction" note="Pretrained normative response" values={map('predicted')} color="diverging" label="TRIBE prediction" />
      <Map title="Deviation" note="Absolute difference" values={map('deviation')} color="deviation" label="absolute deviation" />
    </section>
    <section className="panel flex flex-wrap items-center justify-between gap-4 p-5">
      <div className="max-w-4xl"><h3 className="font-semibold">Plain-language result</h3><p className="mt-1 text-sm leading-relaxed" style={{ color: 'var(--fg-muted)' }}>{done ? "The observed fMRI pattern differs from TRIBE’s average-reference response to this stimulus. The deviation view shows where that difference is relatively greater. It does not establish ADHD, brain injury, a language disorder, or any diagnosis." : 'Available after the analysis finishes.'}</p></div>
      {done ? <a className="primary-button" href={`${apiBase()}/exploratory/runs/${run!.id}/report`}>Download report</a> : <span className="status-pill">Awaiting analysis</span>}
    </section>
    <p className="mx-auto max-w-4xl text-center text-xs leading-relaxed" style={{ color: 'var(--fg-faint)' }}>Free mode: open a Kaggle or Colab notebook with a GPU, upload the selected image plus the downloaded script, run it once, then upload its <code>.npy</code> result here. The notebook receives no BOLD fMRI. Research use only; this is not diagnostic or a production-grade surface analysis.</p>
  </div>;
}

function Map({ title, note, values, color, label }: { title: string; note: string; values: string | null; color: 'diverging' | 'deviation'; label: string }) {
  return <div><div className="mb-2 px-1"><h3 className="font-semibold">{title}</h3><p className="text-xs" style={{ color: 'var(--fg-muted)' }}>{note}</p></div><CorticalViewer valuesUrl={values} colorMap={color} legendLabel={label} height={360} /></div>;
}
