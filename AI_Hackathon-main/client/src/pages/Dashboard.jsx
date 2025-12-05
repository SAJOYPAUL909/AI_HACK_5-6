// src/pages/Dashboard.jsx
import React, { useEffect, useState } from "react";

const API_BASE_URL = "http://localhost:5000";

const Dashboard = () => {
  const [user, setUser] = useState(null);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  useEffect(() => {
    const stored = localStorage.getItem("user");
    if (stored) {
      setUser(JSON.parse(stored));
    }
  }, []);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
    setError("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResult(null);

    if (!file) {
      setError("Please select a PDF file to analyze.");
      return;
    }

    try {
      setLoading(true);
      const formData = new FormData();
      formData.append("file", file);

      const token = localStorage.getItem("token");

      const res = await fetch(`${API_BASE_URL}/api/documents/analyze`, {
        method: "POST",
        headers: {
          // Let browser set Content-Type for FormData; just pass auth
          Authorization: token ? `Bearer ${token}` : "",
        },
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.message || "Failed to analyze document");
      }

      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getClassLabel = () => {
    if (!result) return "";
    return result.isTampered ? "Tampered" : "Not Tampered";
  };

  const getClassPillClass = () => {
    if (!result) return "status-pill";
    return result.isTampered
      ? "status-pill status-bad"
      : "status-pill status-ok";
  };

  const formatScore = (score) => {
    if (score == null) return "-";
    return `${Math.round(score * 100)}%`;
  };

  return (
    <div className="dashboard">
      <section className="dashboard-header">
        <div>
          <h1>Document Screening</h1>
          <p>
            {user
              ? `Hi ${user.name}, upload a document to screen it for potential tampering.`
              : "Upload a document to screen it for potential tampering."}
          </p>
        </div>
      </section>

      <section className="dashboard-grid single-column">
        {/* Upload card */}
        <div className="dashboard-card upload-card">
          <h3>Upload Document</h3>
          <p className="upload-subtitle">
            Supported: PDF. The system will analyze the document and return an
            authenticity score and tampering insights.
          </p>

          {error && <div className="auth-error">{error}</div>}

          <form className="upload-form" onSubmit={handleSubmit}>
            <div className="form-group">
              <label>Select file</label>
              <input
                type="file"
                accept="application/pdf"
                onChange={handleFileChange}
              />
              {file && (
                <div className="file-chip">
                  <span>{file.name}</span>
                </div>
              )}
            </div>

            <button
              className="btn btn-primary"
              type="submit"
              disabled={loading}
            >
              {loading ? "Analyzing..." : "Analyze Document"}
            </button>
          </form>
        </div>

        {/* Results card */}
        {result && (
          <div className="dashboard-card results-card">
            <h3>Analysis Result</h3>

            <div className="results-grid">
              {/* Score + Classification */}
              <div className="result-block">
                <h4>Score</h4>
                <p className="score-value">{formatScore(result.score)}</p>
                <p className="score-caption">
                  Higher score indicates higher confidence in authenticity.
                </p>
              </div>

              <div className="result-block">
                <h4>Classification</h4>
                <div className={getClassPillClass()}>{getClassLabel()}</div>
                <p className="score-caption">
                  Based on combined checks from metadata, visual, and textual
                  cues.
                </p>
              </div>

              {/* Potential areas */}
              <div className="result-block full-width">
                <h4>Potential areas of tamper</h4>
                {result.potentialAreas && result.potentialAreas.length > 0 ? (
                  <ul className="tamper-list">
                    {result.potentialAreas.map((item, idx) => (
                      <li key={idx}>{item}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="muted">
                    No specific tampered areas identified with high confidence.
                  </p>
                )}
              </div>

              {/* Explanation */}
              <div className="result-block full-width">
                <h4>Explanation / Rationale</h4>
                <p className="explanation-text">
                  {result.explanation ||
                    "No detailed explanation provided by backend."}
                </p>
              </div>

              {/* Annotated Image */}
              {result.annotatedImageUrl && (
                <div className="result-block full-width">
                  <h4>Visual Highlights</h4>
                  <p className="muted">
                    The image below shows the suspected tampered regions
                    outlined.
                  </p>
                  <div className="annotated-image-wrapper">
                    <img
                      src={result.annotatedImageUrl}
                      alt="Annotated document showing tampered areas"
                      className="annotated-image"
                    />
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </section>
    </div>
  );
};

export default Dashboard;
