import React, { useEffect, useState } from "react";
import UploadPanel from "./components/UploadPanel";
import SidebarHistory from "./components/SidebarHistory";
import DocumentViewer from "./components/DocumentViewer";
import ChatPanel from "./components/ChatPanel";
import { getHistory } from "./api";

export default function App(){
  const [history, setHistory] = useState([]);
  const [selected, setSelected] = useState(null);

  async function loadHistory(){
    try {
      const h = await getHistory();
      setHistory(h);
    } catch(e) {
      console.error(e);
    }
  }

  useEffect(()=>{
    loadHistory();
  },[]);

  return (
    <div className="app-root">
      <div className="left-col">
        <UploadPanel onUploaded={loadHistory} />
        <SidebarHistory history={history} onSelect={setSelected} />
      </div>
      <div className="center-col">
        <DocumentViewer document={selected} onRefresh={loadHistory} />
      </div>
      <div className="right-col">
        <ChatPanel document={selected} />
      </div>
    </div>
  );
}
