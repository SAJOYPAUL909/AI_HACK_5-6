// controllers/documentController.js
import fs from "fs";
import path from "path";
import axios from "axios";
import FormData from "form-data";

const PYTHON_SERVICE_URL = "http://localhost:8000/analyze";

export const analyzeDocument = async (req, res, next) => {
  try {
    if (!req.file) {
      return res.status(400).json({ message: "No file uploaded" });
    }

    const pdfPath = req.file.path;

    // Prepare FormData to send to Python service
    const formData = new FormData();
    formData.append("file", fs.createReadStream(pdfPath), {
      filename: path.basename(pdfPath),
      contentType: "application/pdf",
    });

    const pythonResponse = await axios.post(PYTHON_SERVICE_URL, formData, {
      headers: {
        ...formData.getHeaders(),
      },
      timeout: 60000,
    });

    // const data = pythonResponse.data;

    // // Python returns annotatedImagePath (relative). We'll turn it into a URL.
    // let annotatedImageUrl = null;
    // if (data.annotatedImagePath) {
    //   // Option 1: serve image from Node (copy file to Node uploads) – more complex
    //   // Option 2: directly serve from Python (if Python exposes static)
    //   // For now, let's assume Node will expose the same /uploads directory.
    //   const fileName = path.basename(data.annotatedImagePath);
    //   annotatedImageUrl = `${req.protocol}://${req.get("host")}/uploads/${fileName}`;
    // }

    // return res.json({
    //   score: data.score ?? 0.5,
    //   isTampered: data.isTampered ?? false,
    //   potentialAreas: data.potentialAreas ?? [],
    //   explanation:
    //     data.explanation ||
    //     "The document was analyzed by the PDF microservice.",
    //   annotatedImageUrl,
    // });
    const data = pythonResponse.data;

const annotatedImageUrl = data.annotatedImageUrl || null;

return res.json({
  score: data.score ?? 0.5,
  isTampered: data.isTampered ?? false,
  potentialAreas: data.potentialAreas ?? [],
  explanation:
    data.explanation || "The document was analyzed by the PDF microservice.",
  annotatedImageUrl, // 👈 just forward the URL from Python
});

  } catch (err) {
    console.error("Error calling Python service:", err.message);
    if (err.response?.data) {
      console.error("Python error response:", err.response.data);
    }
    return res
      .status(500)
      .json({ message: "Failed to analyze document via microservice" });
  }
};
