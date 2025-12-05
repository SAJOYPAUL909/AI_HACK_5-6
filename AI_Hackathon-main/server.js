// server.js
import express from "express";
import dotenv from "dotenv";
import cors from "cors";
import morgan from "morgan";
import path from "path";

import { sequelize } from "./models/index.js";
import authRoutes from "./routes/authRoutes.js";
import { notFound, errorHandler } from "./middleware/errorMiddleware.js";
import documentRoutes from "./routes/documentRoutes.js";

import { fileURLToPath } from "url";


dotenv.config();

const app = express();

// Middlewares
app.use(cors());
app.use(express.json());
app.use(morgan("dev"));

// Routes
app.get("/", (req, res) => {
  res.json({ message: "Document Screening API (MySQL) is running" });
});

app.use("/api/auth", authRoutes);
app.use("/uploads", express.static(path.join(process.cwd(), "uploads")));
app.use("/api/documents", documentRoutes);


// Error handlers
app.use(notFound);
app.use(errorHandler);

// Sync DB and start server
const PORT = process.env.PORT || 5000;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ...
app.use("/uploads", express.static(path.join(__dirname, "uploads")));

sequelize
  .sync() // use { alter: true } during dev if you want auto-migrations
  .then(() => {
    console.log("MySQL synced successfully");
    app.listen(PORT, () =>
      console.log(`Server running on http://localhost:${PORT}`)
    );
  })
  .catch((err) => {
    console.error("DB sync error:", err.message);
  });
