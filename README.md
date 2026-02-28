# 3D Coaster Generator

Transform any image into a 3D printable coaster using AI-powered image processing and vector-based 3D modeling.

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Three.js](https://img.shields.io/badge/Three.js-black?style=for-the-badge&logo=three.js&logoColor=white)](https://threejs.org/)

## 🎯 Features

- **AI-Powered Image Processing**: Uses BFL FLUX.2 API to convert images into clean, vector-ready graphics
- **User Authentication**: Google OAuth integration for user accounts and session tracking
- **Usage Quotas & Tiers**: 
  - Anonymous users: 1 free generation
  - Signed-in users: 2 additional free generations
  - Paid users: High monthly limits (configurable)
- **Monetization Ready**: Integrated with Dodo Payments for easy subscription upgrades
- **Interactive Review Workflow**: Preview AI-generated images before 3D conversion
- **Real-time 3D Preview**: Interactive Three.js viewer to visualize coasters before download
- **Combined 3MF Output**: Single 3MF containing body + logos for multi-color printing

## 🏗️ Architecture

### Pipeline Workflow

1. **Quota Check** → Ensure user has remaining credits
2. **Image Upload** → User uploads image via web interface
3. **AI Processing** → BFL FLUX.2 generates clean black & white vector-style image
4. **Review Phase** → User reviews and approves the generated image
5. **Vectorization** → vtracer converts image to SVG paths
6. **3D Generation** → trimesh creates extruded 3D models
7. **Download** → Single 3MF file with body + logos ready for printing

### Tech Stack

**Backend:**
- FastAPI (async web framework)
- SQLite (auth, quotas, billing persistence)
- Authlib (Google OAuth)
- Dodo Payments Python SDK
- BFL API (FLUX.2 Klein 9B for image processing)
- vtracer (SVG vectorization)
- trimesh (3D mesh generation)

**Frontend:**
- Vanilla JavaScript
- Tailwind CSS
- Three.js (3D STL viewer)

## 📋 Prerequisites

- Python 3.8+
- BFL API Key ([Get one here](https://api.bfl.ai))
- Google OAuth Credentials (for authentication)
- Dodo Payments API Key (for paid tier)
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

Create a `.env` file in the root directory:
```bash
# Core
ENVIRONMENT=development
BFL_API_KEY=your_bfl_api_key_here

# Auth / Sessions
SESSION_SECRET=your_super_secret_random_string
OAUTH_GOOGLE_CLIENT_ID=your_google_client_id
OAUTH_GOOGLE_CLIENT_SECRET=your_google_client_secret
PUBLIC_BASE_URL=http://localhost:8000

# Quotas
ANON_FREE_LIMIT=1
LOGIN_BONUS_LIMIT=2
PAID_MONTHLY_LIMIT=200

# Billing (Optional - via Dodo Payments)
DODO_PAYMENTS_API_KEY=your_dodo_api_key
DODO_PAYMENTS_ENVIRONMENT=test_mode
DODO_PAYMENTS_WEBHOOK_KEY=your_dodo_webhook_secret
DODO_SUBSCRIPTION_PRODUCT_ID=your_product_id
```

### 5. Run the Application

```bash
python main.py
# Or use uvicorn directly:
# uvicorn main:app --host 0.0.0.0 --port 8000
```

The application will be available at `http://localhost:8000`

## ⚙️ Usage

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

### Webhook Setup (For Payments)
If using the paid tier, configure your Dodo Payments webhook to point to:
`https://your-domain.com/api/billing/webhook`

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

## 🚀 Deployment

### Nixpacks / Coolify Deployment
This project includes a `nixpacks.toml` configured for easy deployment on platforms like Coolify or Railway.

Ensure you map a **Persistent Volume** to `/app/temp` to preserve the SQLite database (`app_data.db`) across deployments.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details
