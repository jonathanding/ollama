import { useState, useEffect } from 'react';
import type { SummaryData, CompareData } from '../types/trace';

interface FileEntry { name: string; size: number; type: 'summary' | 'compare'; }

const BASE = import.meta.env.DEV ? 'http://localhost:8765' : '';

export function useFileList() {
  const [files, setFiles] = useState<FileEntry[]>([]);
  useEffect(() => {
    fetch(`${BASE}/api/files`).then(r => r.json()).then(setFiles).catch(() => {});
  }, []);
  return files;
}

export function useSummary(filename: string | null) {
  const [data, setData] = useState<SummaryData | null>(null);
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    if (!filename) return;
    setLoading(true);
    fetch(`${BASE}/data/${filename}`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, [filename]);
  return { data, loading };
}

export function useCompare(filename: string | null) {
  const [data, setData] = useState<CompareData | null>(null);
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    if (!filename) return;
    setLoading(true);
    fetch(`${BASE}/data/${filename}`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, [filename]);
  return { data, loading };
}
