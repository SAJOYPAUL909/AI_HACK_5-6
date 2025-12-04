import React, { useState } from "react";
import { chatDoc } from "../api";

export default function ChatPanel({ document }) {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);

  async function handleAsk(){
    if (!document) return alert("Select a document");
    if (!question) return;
    const q = question;
    setMessages(prev => [...prev, { from: "user", text: q }]);
    setQuestion("");
    try {
      const res = await chatDoc(document.id, q);
      setMessages(prev => [...prev, { from: "assistant", text: res.answer }]);
    } catch(e){
      setMessages(prev => [...prev, { from: "assistant", text: "Error from server" }]);
    }
  }

  return (
    <div className="chat-panel">
      <h4>Ask AI</h4>
      <div className="messages">
        {messages.map((m, idx)=> (
          <div key={idx} className={`msg ${m.from}`}>{m.text}</div>
        ))}
      </div>
      <div className="input-row">
        <input value={question} onChange={e=>setQuestion(e.target.value)} placeholder="Ask about the document..." />
        <button onClick={handleAsk}>Send</button>
      </div>
    </div>
  );
}
