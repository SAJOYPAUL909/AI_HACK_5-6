// services/documentAnalysisService.js
import  { PDFParse } from "pdf-parse";
import fs from "fs";
import path from "path";
import sharp from "sharp";
import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

/**
 * STEP 1: Extract text + metadata from PDF (using pdf-parse).
 * For full OCR + bounding boxes, you'd integrate Tesseract here.
 */
export const extractTextAndMetadata = async (pdfBuffer) => {
  const data = await PDFParse(pdfBuffer);  // <--- correct usage

  return {
    text: data.text || "",
    metadata: {
      title: data.info?.Title || null,
      author: data.info?.Author || null,
      creator: data.info?.Creator || null,
      producer: data.info?.Producer || null,
      creationDate: data.info?.CreationDate || null,
      modDate: data.info?.ModDate || null,
    },
  };
};


/**
 * STEP 2: VERY SIMPLE rule-based checks (placeholder for region-based analysis)
 * Here you can later plug in:
 * - template-based region bounding boxes
 * - expected length checks for dates/amounts
 * - spacing anomalies
 */
export const runRuleBasedChecks = ({ text, metadata }) => {
  const issues = [];

  // Example: check for prohibited words
  const lower = text.toLowerCase();
  const forbiddenWords = ["guaranteed", "guarantee", "assured return", "fixed double"];
  const matchedForbidden = forbiddenWords.filter((w) => lower.includes(w));

  if (matchedForbidden.length > 0) {
    issues.push(
      `Document contains potentially prohibited marketing terms: ${matchedForbidden.join(
        ", "
      )}.`
    );
  }

  // Example: very rough date heuristic (you can improve this with regex / region extraction)
  const dateRegex = /\b(\d{1,2}[-\/]\d{1,2}[-\/]\d{2,4})\b/g;
  const dates = text.match(dateRegex) || [];
  if (dates.length > 1) {
    issues.push(`Multiple dates detected in document: ${dates.slice(0, 5).join(", ")}.`);
  }

  // Metadata heuristic
  if (metadata.creationDate && metadata.modDate) {
    if (metadata.modDate !== metadata.creationDate) {
      issues.push(
        `PDF metadata shows different creation (${metadata.creationDate}) and modification (${metadata.modDate}) dates.`
      );
    }
  }

  return { issues, dates, matchedForbidden };
};

/**
 * STEP 3: Image / "forensics" placeholder.
 * Here you'd convert first page to image and run:
 * - Error Level Analysis
 * - Noise / compression differences
 * For now we just export the first page as a PNG "annotated" placeholder.
 */
export const generateAnnotatedPreview = async (pdfPath) => {
  // Naive: just copy the PDF as PNG-ish placeholder using sharp.
  // In a real implementation:
  //  - Use 'pdf-poppler' or 'pdftoppm' to get a proper rasterized page,
  //  - Then draw rectangles around suspicious regions.
  const outputName = `annotated_${path.basename(pdfPath, path.extname(pdfPath))}.png`;
  const outputPath = path.join(path.dirname(pdfPath), outputName);

  try {
    // Sharp can rasterize PDFs if underlying system has libvips compiled with PDF support.
    // If this fails, you can skip or use external poppler tools.
    await sharp(pdfPath, { density: 150 }).png().toFile(outputPath);
    return outputPath;
  } catch (err) {
    console.error("Preview generation failed:", err.message);
    return null;
  }
};

/**
 * STEP 4: Use OpenAI for semantic consistency + final structure
 * We ask it to return STRICT JSON with our desired fields.
 */
export const runLlmAnalysis = async ({ text, metadata, ruleIssues }) => {
  // Build a compact metadata summary
  const metadataSummary = JSON.stringify(metadata, null, 2);
  const ruleSummary = ruleIssues.length
    ? ruleIssues.map((x, i) => `${i + 1}. ${x}`).join("\n")
    : "No obvious rule-based issues detected.";

  const prompt = `
You are a document forensics assistant. You receive:
- Extracted plain text of a filled form or certificate
- Basic PDF metadata
- Simple rule-based anomaly flags

You must decide:
1. An authenticity score between 0 and 1 (1 = very authentic, 0 = clearly tampered).
2. Whether the document is tampered or not (true/false).
3. A list of potential areas of tamper (high-level descriptions, not coordinates).
4. A clear explanation (2–5 sentences) of the rationale for identifying such tampering.

Return ONLY valid JSON in this exact format:

{
  "score": 0.0,
  "isTampered": true,
  "potentialAreas": ["..."],
  "explanation": "..."
}

Now analyze:

[PDF METADATA]
${metadataSummary}

[RULE-BASED ISSUES]
${ruleSummary}

[DOCUMENT TEXT]
${text.slice(0, 8000)}
  `.trim();

  const response = await openai.responses.create({
    model: "gpt-4o-mini", // or gpt-5, depending on your plan
    input: prompt,
    max_output_tokens: 400,
    text: { format: { type: "text" } },
  });

  // For JS SDK, we can use output_text convenience:
  const raw = response.output_text || response.output?.[0]?.content?.[0]?.text || "";
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (err) {
    console.error("Failed to parse LLM JSON, raw:", raw);
    // Fallback minimal structure
    parsed = {
      score: 0.5,
      isTampered: false,
      potentialAreas: [],
      explanation:
        "Unable to parse structured output from the model. Please review the document manually.",
    };
  }

  return parsed;
};
