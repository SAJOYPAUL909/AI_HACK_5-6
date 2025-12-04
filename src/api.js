const BASE = process.env.REACT_APP_API_BASE || "http://localhost:8000/api";

export async function uploadFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${BASE}/upload`, {
    method: "POST",
    body: fd
  });
  return res.json();
}

export async function analyzeDoc(id) {
  const res = await fetch(`${BASE}/analyze/${id}`, { method: "POST" });
  return res.json();
}

export async function getHistory() {
  const res = await fetch(`${BASE}/history`);
  return res.json();
}

export async function chatDoc(id, question) {
  const res = await fetch(`${BASE}/chat/${id}`, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ question })
  });
  return res.json();
}

export async function downloadReport(id) {
  const res = await fetch(`${BASE}/report/${id}`);
  if (!res.ok) throw new Error("report failed");
  const blob = await res.blob();
  return blob;
}
