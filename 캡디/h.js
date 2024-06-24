const express = require('express');
const multer = require('multer');
const fs = require('fs');

const app = express();
const port = 3000;

app.use(express.static('public'));

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'public/uploads/');
  },
  filename: (req, file, cb) => {
    cb(null, file.originalname);
  }
});

const upload = multer({ storage });

app.post('/upload', upload.single('file'), (req, res) => {
  const filePath = `public/uploads/${req.file.originalname}`;
  const fileContent = fs.readFileSync(filePath, 'utf-8');
  res.send(fileContent);
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});