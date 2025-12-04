import React, { useState } from "react";
import { uploadFile } from "../api";

export default function UploadPanel({ onUploaded }) {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");

  async function handleUpload(e) {
    e.preventDefault();
    if (!file) return setMessage("Select a file");
    try {
      setMessage("Uploading...");
      const res = await uploadFile(file);
      if (res.document_id) {
        setMessage("Uploaded. Analysis available from dashboard.");
        setFile(null);
        onUploaded && onUploaded();
      } else {
        setMessage("Upload failed");
      }
    } catch (err) {
      console.error(err);
      setMessage("Error uploading");
    }
  }

  return (
    <div className="upload-panel">
      <h3>Upload Document</h3>
      <form onSubmit={handleUpload}>
        <input type="file" onChange={e=>setFile(e.target.files[0])} />
        <button type="submit">Upload</button>
      </form>
      <div>{message}</div>
    </div>
  );
}
