# ============================================================
#  HMF-m6A  m6A Methylation Predictor-Smart Launcher
#  Features: Auto-detect environment, check packages, auto-install or update
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "  ====================================================="
echo "    HMF-m6A  m6A Methylation Predictor  -  Web App"
echo "  ====================================================="
echo ""

# ============================================================
#  Step 1: Check Python availability
# ============================================================
if command -v python3.9 &>/dev/null; then
    PYTHON_CMD=python3.9
elif command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
else
    echo "  [ERROR] Python 3 is not found!"
    echo ""
    echo "  Please install Python 3.9 or higher first."
    echo ""
    read -p "  Press Enter to exit..."
    exit 1
fi
echo "  [OK] Python detected."

# ============================================================
#  Step 2: Check if virtual environment exists and verify packages
# ============================================================
VENV_DIR="$SCRIPT_DIR/.venv"
NEEDS_UPDATE=false
NEEDS_INSTALL=false

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/python" ]; then
    echo "  [OK] Virtual environment detected."
    
    source "$VENV_DIR/bin/activate"
    
    echo ""
    echo "  Checking package versions..."
    
    # Check torch
    TORCH_VER=$($PYTHON_CMD -c "import torch; print(torch.__version__)" 2>/dev/null || echo "NOT_INSTALLED")
    if [ "$TORCH_VER" = "NOT_INSTALLED" ]; then
        echo "  [UPDATE] PyTorch not found"
        NEEDS_UPDATE=true
    else
        echo "  [OK] PyTorch $TORCH_VER"
    fi
    
    # Check critical packages
    for PKG in streamlit numpy pandas transformers scipy scikit-learn matplotlib seaborn plotly kaleido; do
        if ! $PYTHON_CMD -c "import $PKG" &>/dev/null; then
            echo "  [UPDATE] $PKG not found"
            NEEDS_UPDATE=true
        else
            VER=$($PYTHON_CMD -c "import $PKG; print($PKG.__version__)" 2>/dev/null || echo "ok")
            echo "  [OK] $PKG $VER"
        fi
    done
    
    # Check ViennaRNA (import name is RNA)
    if ! $PYTHON_CMD -c "import RNA" &>/dev/null; then
        echo "  [UPDATE] ViennaRNA not found"
        NEEDS_UPDATE=true
    else
        echo "  [OK] ViennaRNA installed"
    fi
    
else
    echo "  [NEW] Virtual environment not found, will create new one."
    NEEDS_INSTALL=true
fi

# ============================================================
#  Step 3: Install or update packages if needed
# ============================================================
if [ "$NEEDS_INSTALL" = true ] || [ "$NEEDS_UPDATE" = true ]; then
    
    if [ "$NEEDS_INSTALL" = true ]; then
        echo ""
        echo "  Creating isolated virtual environment in: .venv/"
        echo "  This will NOT affect any system Python or other environments."
        echo ""
        "$PYTHON_CMD" -m venv "$VENV_DIR"
        source "$VENV_DIR/bin/activate"
        pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
    fi
    
    echo ""
    echo "  -------------------------------------------------------"
    echo "  Installing/updating packages (this may take a few minutes)..."
    echo "  -------------------------------------------------------"
    
    # Install PyTorch 2.5.1 with CUDA 12.1 (same as conda m6A environment)
    echo ""
    echo "  Installing PyTorch 2.5.1 with CUDA 12.1..."
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    if [ $? -ne 0 ]; then
        echo "  [WARNING] CUDA PyTorch failed, falling back to CPU..."
        pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
        if [ $? -ne 0 ]; then
            echo "  [ERROR] Failed to install PyTorch!"
            read -p "  Press Enter to exit..."
            exit 1
        fi
    fi
    
    # Install other dependencies (all pinned to match conda m6A environment)
    echo ""
    echo "  Installing other dependencies..."
    pip install -r "$SCRIPT_DIR/requirements_venv.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple
    if [ $? -ne 0 ]; then
        echo "  [WARNING] Some packages may not install correctly."
        echo "  Continuing anyway..."
    fi
    
    echo ""
    echo "  ====================================================="
    echo "    Installation/update complete!"
    echo "  ====================================================="
    echo ""
    
fi

# ============================================================
#  Step 4: Ensure environment is activated
# ============================================================
if [ -z "$VIRTUAL_ENV" ]; then
    source "$VENV_DIR/bin/activate"
fi

# ============================================================
#  Step 5: Launch the Streamlit web application
# ============================================================
echo "  Starting HMF-m6A Web Application ..."
echo ""
echo "  The app will be available at:"
    echo "    http://localhost:8502"
echo ""
echo "  Press Ctrl+C to stop the server."
echo "  ====================================================="
echo ""

export MPLCONFIGDIR="/tmp/matplotlib_cache_$$"
mkdir -p "$MPLCONFIGDIR"

streamlit run app.py --server.port 8502

if [ $? -ne 0 ]; then
    echo ""
    echo "  [ERROR] Application failed to start!"
    echo "  Please check the error messages above."
    echo ""
    read -p "  Press Enter to exit..."
fi
