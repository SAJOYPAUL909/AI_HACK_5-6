// // routes/documentRoutes.js
// import express from "express";
// import { analyzeDocument } from "../controllers/documentController.js";
// import { upload } from "../middleware/uploadMiddleware.js";
// import { protect } from "../middleware/authMiddleware.js";

// const router = express.Router();

// // Protected route: user must be logged in
// router.post("/analyze", protect, upload.single("file"), analyzeDocument);

// export default router;


// routes/documentRoutes.js
import express from "express";
import { analyzeDocument } from "../controllers/documentController.js";
import { upload } from "../middleware/uploadMiddleware.js";
import { protect } from "../middleware/authMiddleware.js";

const router = express.Router();

router.post("/analyze", protect, upload.single("file"), analyzeDocument);

export default router;
