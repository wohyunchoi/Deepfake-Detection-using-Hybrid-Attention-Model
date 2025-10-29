import React, { useState } from "react";

function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [isDragging, setIsDragging] = useState(false);

  const handleUpload = async () => {
    if (!image) return;
    setLoading(true);
    setResult(null);
    setProgress(20);

    const formData = new FormData();
    formData.append("file", image);

    try {
      setProgress(50);
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });
      setProgress(80);

      const data = await res.json();
      setResult(data);
      setProgress(100);
    } catch (err) {
      console.error(err);
      setResult({ error: "Upload failed" });
    }

    setTimeout(() => {
      setLoading(false);
      setProgress(0);
    }, 1000);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      setImage(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const renderConfidenceBar = (confidence, label) => {
    const percent = Math.round(confidence * 100);
    const color = label === "deepfake" ? "#ff4d4d" : "#4CAF50";

    return (
      <div style={{ width: "100%", marginTop: 10 }}>
        <div
          style={{
            height: "15px",
            background: "#ddd",
            borderRadius: "8px",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              width: `${percent}%`,
              background: color,
              height: "100%",
              transition: "width 0.5s",
            }}
          ></div>
        </div>
        <p style={{ fontSize: "14px", marginTop: 5 }}>
          Confidence: {percent}%
        </p>
      </div>
    );
  };

  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        minHeight: "100vh",
        background: "linear-gradient(135deg, #1e1f26, #3a3b47)",
        color: "white",
        fontFamily: "Arial, sans-serif",
      }}
    >
      <div
        style={{
          background: "#2b2c36",
          padding: 30,
          borderRadius: 15,
          boxShadow: "0 0 15px rgba(0,0,0,0.3)",
          width: "400px",
          textAlign: "center",
        }}
      >
        <h2 style={{ marginBottom: 20 }}>Defake: Deepfake Detection using Hybird Attention Model</h2>

        {/* Drag & Drop */}
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          style={{
            border: isDragging ? "2px dashed #00bcd4" : "2px dashed #555",
            background: isDragging ? "#3a3b47" : "#2b2c36",
            padding: "20px",
            borderRadius: "10px",
            marginBottom: "10px",
            transition: "0.3s",
            cursor: "pointer",
          }}
        >
          {image ? (
            <p>{image.name}</p>
          ) : (
            <p>Drag & Drop image here or click below</p>
          )}
        </div>

        <input
          type="file"
          accept="image/*"
          onChange={(e) => setImage(e.target.files[0])}
          style={{
            marginBottom: 10,
            background: "#3f404d",
            border: "none",
            color: "white",
            padding: "10px",
            borderRadius: "8px",
          }}
        />
        <br />
        <button
          onClick={handleUpload}
          disabled={loading}
          style={{
            background: loading ? "#555" : "#4CAF50",
            color: "white",
            padding: "10px 20px",
            border: "none",
            borderRadius: "8px",
            cursor: loading ? "not-allowed" : "pointer",
            transition: "0.3s",
          }}
        >
          {loading ? "Analyzing..." : "Upload"}
        </button>

        {progress > 0 && loading && (
          <div
            style={{
              height: "6px",
              width: "100%",
              background: "#444",
              borderRadius: "4px",
              marginTop: "15px",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                width: `${progress}%`,
                background: "#00bcd4",
                height: "100%",
                transition: "width 0.3s ease",
              }}
            ></div>
          </div>
        )}

        {image && (
          <div style={{ marginTop: 20 }}>
            <img
              src={URL.createObjectURL(image)}
              alt="preview"
              style={{
                maxWidth: "100%",
                borderRadius: "10px",
                boxShadow: "0 0 10px rgba(0,0,0,0.4)",
              }}
            />
          </div>
        )}

        {result && !result.error && (
          <div style={{ marginTop: 20 }}>
            <h3
              style={{
                color: result.label === "deepfake" ? "#ff4d4d" : "#4CAF50",
                textTransform: "uppercase",
                marginBottom: 10,
              }}
            >
              {result.label}
            </h3>
            {renderConfidenceBar(result.confidence, result.label)}
          </div>
        )}

        {result && result.error && (
          <p style={{ color: "red", marginTop: 15 }}>{result.error}</p>
        )}
      </div>
    </div>
  );
}

export default App;