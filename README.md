# 3D Coaster Generator

Transform any image into a 3D printable coaster using AI-powered image processing and vector-based 3D modeling.

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Three.js](https://img.shields.io/badge/Three.js-black?style=for-the-badge&logo=three.js&logoColor=white)](https://threejs.org/)

## 🎯 Features

- **AI-Powered Image Processing**: Uses BFL FLUX.2 API to convert images into clean, vector-ready graphics
- **Interactive Review Workflow**: Preview AI-generated images before 3D conversion
- **Real-time 3D Preview**: Interactive Three.js viewer to visualize coasters before download
- **Combined 3MF Output**: Single 3MF containing body + logos for multi-color printing
- **Self-Serve API Keys**: Users can optionally provide their own BFL API keys
- **Responsive Web Interface**: Modern, mobile-friendly UI with progress tracking
- **Configurable Parameters**: Full control over dimensions, logo depth, scaling, and rotation

## 🏗️ Architecture

### Pipeline Workflow

1. **Image Upload** → User uploads image via web interface
2. **AI Processing** → BFL FLUX.2 generates clean black & white vector-style image
3. **Review Phase** → User reviews and approves the generated image
4. **Vectorization** → vtracer converts image to SVG paths
5. **3D Generation** → trimesh creates extruded 3D models
6. **Download** → Single 3MF file with body + logos ready for printing

### Tech Stack

**Backend:**
- FastAPI (async web framework)
- BFL API (FLUX.2 Klein 9B for image processing)
- vtracer (SVG vectorization)
- trimesh (3D mesh generation)
- matplotlib (preview generation)

**Frontend:**
- Vanilla JavaScript
- Three.js (3D STL viewer)
- Responsive CSS

**Storage:**
- In-memory job tracking
- Temporary file storage

## 📋 Prerequisites

- Python 3.8+
- BFL API Key ([Get one here](https://api.bfl.ai))
- 4GB+ RAM recommended
- 2+ CPU cores recommended

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/bagdeabhishek/coaster-generator.git
cd coaster-generator
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
export BFL_API_KEY="your_bfl_api_key_here"
```

Or create a `.env` file:
```
BFL_API_KEY=your_bfl_api_key_here
```

### 5. Run the Application

```bash
python main.py
```

The application will be available at `http://localhost:8000`

## 💻 Usage

### Web Interface

1. **Upload Image**: Select an image with clear, high-contrast subjects
2. **Configure Parameters**:
   - **Diameter**: Coaster size (50-200mm, default: 100mm)
   - **Thickness**: Coaster height (2-20mm, default: 5mm)
   - **Logo Depth**: Extrusion depth for logos (0.1-5mm, default: 0.6mm)
   - **Scale**: Logo size relative to coaster (0.1-1.5, default: 0.85)
   - **Flip Horizontal**: Mirror the design
   - **Top/Bottom Rotation**: Rotate logos on each side
3. **Review**: Approve or retry the AI-generated image
4. **3D Preview**: Interact with the 3D model (rotate, zoom, pan)
5. **Download**: Get your combined 3MF file

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/process` | POST | Start new coaster job |
| `/api/status/{job_id}` | GET | Check job status |
| `/api/confirm/{job_id}` | POST | Confirm and proceed to 3D |
| `/api/retry/{job_id}` | POST | Retry with new image |
| `/api/preview-image/{job_id}` | GET | Get generated image |
| `/api/download/{job_id}` | GET | Download combined 3MF |
| `/api/download/{job_id}/body` | GET | Download Body STL (viewer) |
| `/api/download/{job_id}/logos` | GET | Download Logos STL (viewer) |
| `/api/download/{job_id}/preview` | GET | Download PNG preview |

### Using Custom API Keys

Users can optionally provide their own BFL API key in the web form. This is useful for:
- Personal usage tracking
- Avoiding shared rate limits
- Separate billing

If no key is provided, the server's environment variable `BFL_API_KEY` is used.

## ⚙️ Configuration

### Backend Configuration

Edit these values in `main.py`:

```python
# Debug mode - keeps temp files for troubleshooting
DEBUG_NO_CLEANUP = False  # Set to True for debugging

# Temp directory location
TEMP_DIR = os.path.abspath("./temp")

# BFL API settings
BFL_API_URL = "https://api.bfl.ai/v1"
MAX_POLLING_ATTEMPTS = 60
POLLING_INTERVAL = 2
```

### Prompt Customization

Edit `prompt.txt` to customize the AI image generation prompt:

```
First, establish the layout: A professional circular stamp emblem defined by two thick, 
concentric black rings...
```

## 🖨️ 3D Printing Guide

### Recommended Settings

- **Material**: PLA or PETG
- **Layer Height**: 0.2mm
- **Infill**: 15-20% for body
- **Supports**: Not required for coasters

### Printing Options

**Single Color:**
- Print the combined `coaster_*.3mf` with a single filament

**Dual Color (Recommended):**
- Open the combined `coaster_*.3mf` in your slicer and assign colors per part
- Or pause and swap filament at the logo layer if your slicer doesn’t support part colors

**Manual Assembly:**
- Use the STL endpoints to print body and logos separately
- Glue logos into the recessed areas

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV BFL_API_KEY=${BFL_API_KEY}
EXPOSE 8000

CMD ["python", "main.py"]
```

### Cloud Deployment

**Recommended Specs:**
- CPU: 2-4 cores
- RAM: 4-8 GB
- Storage: 20 GB SSD
- OS: Ubuntu 20.04+

**Platform Options:**
- **DigitalOcean Droplet**: $12-24/month
- **AWS EC2 t3.medium**: ~$30/month
- **Linode**: $10-20/month
- **Google Cloud Run**: Pay-per-use

### Environment Variables for Production

```bash
export BFL_API_KEY="your_key"
export DEBUG_NO_CLEANUP="false"
export TEMP_DIR="/tmp/coaster-temp"
```

## 🔧 Troubleshooting

### Common Issues

**"No BFL API key available"**
- Set `BFL_API_KEY` environment variable
- Or provide key in web form

**Vectorization fails**
- Ensure uploaded image is valid PNG/JPEG
- Check temp directory has write permissions

**3D generation produces empty STL**
- SVG may have no closed paths
- Try different image with clearer subjects

**Server won't start**
- Check port 8000 is not in use
- Verify all dependencies installed

### Debug Mode

Enable debug mode to preserve all temp files:

```python
DEBUG_NO_CLEANUP = True  # In main.py
```

This saves intermediate files for inspection:
- `temp_xxx.png` - BFL generated image
- `temp_xxx.svg` - Vectorized SVG
- `xxx_debug.svg` - Debug SVG output
- `xxx_Body.stl` - Base cylinder
- `xxx_Logos.stl` - Extruded logos

## 📊 Performance

### Processing Times (Approximate)

| Phase | Duration | Resource Usage |
|-------|----------|----------------|
| BFL API Call | 5-30s | Network I/O |
| Vectorization | 2-5s | CPU (single core) |
| 3D Generation | 10-30s | CPU + RAM |
| Preview Generation | 3-8s | CPU + RAM |

### Concurrent Job Limits

- **BFL API**: Depends on your account tier
- **Server**: Limited by CPU cores and RAM
- **Recommendation**: Process 3-5 jobs concurrently max

## 🛡️ Security Considerations

- API keys are stored in memory only (not persisted)
- Temp files are cleaned up after processing (unless DEBUG mode)
- No user authentication implemented (add if needed for production)
- Consider adding rate limiting for public deployments

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- [BFL](https://api.bfl.ai) for the FLUX.2 API
- [trimesh](https://trimsh.org/) for 3D mesh operations
- [vtracer](https://github.com/visioncortex/vtracer) for SVG vectorization
- [Three.js](https://threejs.org/) for web-based 3D visualization
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework

## 📞 Support

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones
- For BFL API issues, contact [BFL Support](https://api.bfl.ai)

---

**Happy 3D Printing! 🎉**
