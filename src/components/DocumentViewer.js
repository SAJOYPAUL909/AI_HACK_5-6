import React, { useEffect, useRef, useState } from "react";
import { analyzeDoc, downloadReport } from "../api";

export default function DocumentViewer({ document, onRefresh }) {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(()=>{
    setAnalysis(null);
  }, [document]);

  if (!document) return <div className="placeholder">Select a document to view</div>;

  async function handleAnalyze(){
    setLoading(true);
    try {
      const res = await analyzeDoc(document.id);
      setAnalysis(res);
      onRefresh && onRefresh();
    } catch(err){
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  async function handleDownload(){
    try {
      const blob = await downloadReport(document.id);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `report_${document.id}.pdf`;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch(e) {
      alert("Failed to download report");
    }
  }

  return (
    <div className="doc-viewer">
      <div className="doc-header">
        <h3>{document.filename}</h3>
        <div>
          <button onClick={handleAnalyze} disabled={loading}>Analyze</button>
          <button onClick={handleDownload}>Download Report</button>
        </div>
      </div>

      <div className="doc-body">
        <div className="score">
          Score: {document.authenticity_score ?? "N/A"} ({document.status})
        </div>
        {analysis && (
          <div className="analysis-box">
            <pre style={{maxHeight:300, overflow:'auto'}}>{JSON.stringify(analysis, null, 2)}</pre>
          </div>
        )}
        <div className="viewer-canvas">
          {/* Placeholder image preview: in prod fetch page image or render PDF canvas */}
          <div style={{border:'1px solid #ddd', padding:12}}>Document preview / annotation canvas will go here.</div>
        </div>
      </div>
    </div>
  );
}
