import React, { useState, useEffect } from 'react';
import { Cpu, CheckCircle2, AlertCircle, HelpCircle, Layers, X } from 'lucide-react';
import { api } from '../services/api';
import { MLStatusResponse } from '../types';

export const MLHealthStatusPill: React.FC = () => {
  const [status, setStatus] = useState<MLStatusResponse | null>(null);
  const [isOpen, setIsOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    api.getMLStatus()
      .then((data) => {
        if (isMounted) {
          setStatus(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err.message || 'Offline');
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  if (loading) {
    return (
      <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-100 border border-slate-200 text-[11px] font-mono text-slate-500 animate-pulse">
        <Cpu className="w-3 h-3 animate-spin text-slate-400" />
        <span>Checking ML Engine...</span>
      </div>
    );
  }

  if (error || !status) {
    return (
      <div 
        className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-rose-50 border border-rose-200 text-[11px] font-mono text-rose-700 cursor-pointer"
        onClick={() => setIsOpen(true)}
        title="ML Engine unavailable"
      >
        <AlertCircle className="w-3 h-3 text-rose-500" />
        <span>ML Offline</span>
      </div>
    );
  }

  const { models, compute_device, pytorch_available } = status;
  const codebertLoaded = models.codebert?.is_available;
  const stLoaded = models.sentence_transformer?.is_available;

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-1 rounded-full bg-slate-50 hover:bg-slate-100 border border-slate-200 text-xs font-mono text-slate-700 transition-colors shadow-sm select-none"
        title="Click to view ML Models & Fallback Architecture status"
      >
        <span className="flex h-2 w-2 relative">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
          <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
        </span>
        <span className="font-bold text-slate-800">ML:</span>
        <span className="text-[11px] font-semibold text-slate-600">
          PyTorch ({compute_device.toUpperCase()})
        </span>
        <span className="text-[10px] text-slate-400">|</span>
        <span className={`text-[10px] px-1.5 py-0.2 rounded font-bold ${
          codebertLoaded ? 'bg-emerald-100 text-emerald-800' : 'bg-slate-200 text-slate-700'
        }`}>
          {codebertLoaded ? 'CodeBERT' : 'AST Fallback'}
        </span>
        <span className={`text-[10px] px-1.5 py-0.2 rounded font-bold ${
          stLoaded ? 'bg-indigo-100 text-indigo-800' : 'bg-slate-200 text-slate-700'
        }`}>
          {stLoaded ? 'ST Embedder' : 'Rubric Fallback'}
        </span>
      </button>

      {/* Popover Detail Modal */}
      {isOpen && (
        <div className="absolute right-0 mt-2 w-80 sm:w-96 bg-white border border-slate-200 rounded-2xl shadow-xl p-5 z-50 text-xs space-y-4 animate-in fade-in zoom-in-95 duration-150">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-sky-600" />
              <h4 className="font-bold text-slate-900 text-sm">Machine Learning System Health</h4>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              className="p-1 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-600"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <div className="space-y-3 font-sans">
            {/* Compute Device */}
            <div className="flex items-center justify-between p-2.5 bg-slate-50 rounded-xl border border-slate-200">
              <span className="font-semibold text-slate-700">Compute Backend:</span>
              <span className="font-mono text-xs font-bold text-slate-900 uppercase">
                {compute_device} {compute_device === 'mps' ? '(Apple Silicon Metal)' : ''}
              </span>
            </div>

            {/* PyTorch Transformer Model */}
            <div className="p-3 bg-slate-50 rounded-xl border border-slate-200 space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="font-bold text-slate-800">Multimodal Fusion Transformer</span>
                <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                  pytorch_available ? 'bg-emerald-100 text-emerald-800' : 'bg-rose-100 text-rose-800'
                }`}>
                  {pytorch_available ? 'PyTorch Active' : 'Fallback Engine'}
                </span>
              </div>
              <p className="text-[11px] text-slate-500">
                Architecture: {models.multimodal_transformer?.name || 'PyTorchMultimodalTransformer'}
              </p>
              <p className="text-[11px] text-slate-500">
                Cross-Attention: Code Query attends to Viva Explanation tokens with LayerNorm and Dropout.
              </p>
            </div>

            {/* CodeBERT Embedder */}
            <div className="p-3 bg-slate-50 rounded-xl border border-slate-200 space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="font-bold text-slate-800">Code Modality (CodeBERT)</span>
                <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                  codebertLoaded ? 'bg-emerald-100 text-emerald-800' : 'bg-amber-100 text-amber-800'
                }`}>
                  {codebertLoaded ? 'Loaded (768-D)' : 'Fallback Representation'}
                </span>
              </div>
              <p className="text-[11px] text-slate-600">
                Model: <code className="text-slate-800 font-bold">{models.codebert?.name}</code>
              </p>
              <p className="text-[11px] text-slate-500">
                Status: {models.codebert?.status}
              </p>
              {models.codebert?.fallback_feature && (
                <div className="p-2 rounded bg-amber-50 border border-amber-200 text-[10px] text-amber-900">
                  Fallback: Using 16-D deterministic AST structural representation (no silent zero-vectors).
                </div>
              )}
            </div>

            {/* Sentence Transformer Embedder */}
            <div className="p-3 bg-slate-50 rounded-xl border border-slate-200 space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="font-bold text-slate-800">Viva Modality (Sentence-Transformer)</span>
                <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                  stLoaded ? 'bg-indigo-100 text-indigo-800' : 'bg-amber-100 text-amber-800'
                }`}>
                  {stLoaded ? 'Loaded (384-D)' : 'Fallback Representation'}
                </span>
              </div>
              <p className="text-[11px] text-slate-600">
                Model: <code className="text-slate-800 font-bold">{models.sentence_transformer?.name}</code>
              </p>
              <p className="text-[11px] text-slate-500">
                Status: {models.sentence_transformer?.status}
              </p>
              {models.sentence_transformer?.fallback_feature && (
                <div className="p-2 rounded bg-amber-50 border border-amber-200 text-[10px] text-amber-900">
                  Fallback: Using 16-D rubric-calibrated semantic scores with verbal depth features.
                </div>
              )}
            </div>
          </div>

          <div className="text-[10px] text-slate-400 border-t border-slate-100 pt-2 flex items-center justify-between">
            <span>Heavy ML packages optional via requirements-ml.txt</span>
            <button
              onClick={() => setIsOpen(false)}
              className="text-sky-600 font-bold hover:underline"
            >
              Dismiss
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
