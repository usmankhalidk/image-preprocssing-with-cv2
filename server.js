const express = require('express');
const cv = require('opencv4nodejs');
const multer = require('multer');

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

function deskew(mat) {
    const gray = mat.cvtColor(cv.COLOR_BGR2GRAY);
    const blurred = gray.gaussianBlur(new cv.Size(5, 5), 0);
    const edges = blurred.canny(75, 200);
    const lines = edges.houghLinesP(1, Math.PI / 180, 100, 30, 5);
    if (lines.length === 0) return mat;
    const angles = [];
    for (let i = 0; i < lines.length; i++) {
    const { x1, y1, x2, y2 } = lines[i];
    let dx = x2 - x1;
    let dy = y2 - y1;
    if (dx === 0) continue;
    let angle = Math.atan2(dy, dx) * 180 / Math.PI;
    if (Math.abs(angle) > 45) angle -= Math.sign(angle) * 90;
    angles.push(angle);
    }
    if (angles.length === 0) return mat;
    angles.sort((a, b) => a - b);
    const medianAngle = angles[Math.floor(angles.length / 2)];
    if (Math.abs(medianAngle) < 0.5) return mat;
    const h = mat.rows;
    const w = mat.cols;
    const rad = medianAngle * Math.PI / 180;
    const absCos = Math.abs(Math.cos(rad));
    const absSin = Math.abs(Math.sin(rad));
    const newW = Math.floor(h * absSin + w * absCos);
    const newH = Math.floor(h * absCos + w * absSin);
    const center = new cv.Point(w / 2, h / 2);
    const rotMat = cv.getRotationMatrix2D(center, -medianAngle, 1);
    rotMat.set(0, 2, rotMat.at(0, 2) + (newW / 2 - w / 2));
    rotMat.set(1, 2, rotMat.at(1, 2) + (newH / 2 - h / 2));
    const deskewed = mat.warpAffine(rotMat, new cv.Size(newW, newH), cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(255, 255, 255));
    return deskewed;
    }
    /* ------------------------------ Helper: Preprocess ------------------------------ */
    async function preprocessImage(imageData) {
    try {
    let mat = cv.imdecode(Buffer.from(imageData));
    // Resize if too large
    const maxDim = 1600;
    if (mat.cols > maxDim || mat.rows > maxDim) {
    const scale = maxDim / Math.max(mat.cols, mat.rows);
    mat = mat.resize(Math.floor(mat.rows * scale), Math.floor(mat.cols * scale));
    }
    // Convert to grayscale for better OCR contrast
    mat = mat.cvtColor(cv.COLOR_BGR2GRAY);
    // Slight brightness and contrast adjustment (alpha=1.05 for contrast, beta=0.05*255≈12.75 for brightness)
    mat = mat.convertTo(mat.type, 1.05, 12.75);
    // Deskew
    mat = deskew(mat);
    // Compress to <400KB with quality reduction
    const maxSizeKB = 400;
    let quality = 85;
    let params = new cv.Vector();
    params.push_back(cv.IMWRITE_JPEG_QUALITY);
    params.push_back(quality);
    let compressed = cv.imencode('.jpg', mat, params);
    while (compressed.length / 1024 > maxSizeKB && quality > 60) {
    quality -= 5;
    params.delete();
    params = new cv.Vector();
    params.push_back(cv.IMWRITE_JPEG_QUALITY);
    params.push_back(quality);
    compressed = cv.imencode('.jpg', mat, params);
    }
    params.delete();
    return compressed;
    } catch (error) {
    console.error("Preprocessing error:", error);
    return imageData;
    }
    }


app.post('/api/preprocess', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file provided' });
    }

    // Use the existing preprocessImage function
    const processedImage = await preprocessImage(req.file.buffer);
    
    res.setHeader('Content-Type', 'image/jpeg');
    res.send(processedImage);
    
  } catch (error) {
    console.error('Preprocess API error:', error);
    res.status(500).json({ error: error.message });
  }
});

app.listen(3000, () => console.log('Server running on port 3000'));