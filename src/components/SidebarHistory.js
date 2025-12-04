import React from "react";

export default function SidebarHistory({ history = [], onSelect }) {
  return (
    <div className="history-panel">
      <h4>History</h4>
      <ul>
        {history.map(h => (
          <li key={h.id} onClick={() => onSelect(h)}>
            <div className="hist-row">
              <div>{h.filename}</div>
              <div>{h.authenticity_score ?? "-"}</div>
            </div>
            <div className="hist-meta">{h.status}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}
