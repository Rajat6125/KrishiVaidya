document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const previewSection = document.getElementById('previewSection');
    const imagePreview = document.getElementById('imagePreview');
    const removeBtn = document.getElementById('removeBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loadingSection = document.getElementById('loadingSection');
    const resultSection = document.getElementById('resultSection');
    const predictionValue = document.getElementById('predictionValue');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceValue = document.getElementById('confidenceValue');
    const resetBtn = document.getElementById('resetBtn');

    let selectedFile = null;

    // --- Upload Handlers ---

    // Click to upload
    uploadArea.addEventListener('click', () => fileInput.click());

    // Drag and Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // File Input Change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file.');
            return;
        }

        selectedFile = file;

        // Preview Image
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            
            // Switch UI
            uploadArea.classList.add('hidden');
            previewSection.classList.remove('hidden');
            resultSection.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    // --- Actions ---

    // Remove Image
    removeBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        previewSection.classList.add('hidden');
        uploadArea.classList.remove('hidden');
    });

    // Analyze Image (API Call)
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // UI State: Loading
        previewSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');

        // Form Data
        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            // Using relative URL so it works when served from FastAPI
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            
            // Show Results
            showResults(data);

        } catch (error) {
            console.error('Error during prediction:', error);
            alert('An error occurred during analysis. Please try again.');
            
            // Reset to preview on error
            loadingSection.classList.add('hidden');
            previewSection.classList.remove('hidden');
        }
    });

    function showResults(data) {
        loadingSection.classList.add('hidden');
        resultSection.classList.remove('hidden');

        // Update DOM
        predictionValue.textContent = data.prediction || 'Unknown';
        
        // Ensure confidence is between 0 and 100
        const confidence = parseFloat(data.confidence) || 0;
        const clampedConfidence = Math.min(Math.max(confidence, 0), 100);
        
        confidenceValue.textContent = `${clampedConfidence.toFixed(2)}%`;
        
        // Delay bar animation slightly for effect
        setTimeout(() => {
            confidenceBar.style.width = `${clampedConfidence}%`;
            
            // Adjust bar color dynamically based on confidence
            if (clampedConfidence < 50) {
                confidenceBar.style.background = 'var(--danger)';
            } else if (clampedConfidence < 80) {
                confidenceBar.style.background = 'var(--accent)';
            } else {
                confidenceBar.style.background = 'var(--primary)';
            }
        }, 100);
    }

    // Reset Flow
    resetBtn.addEventListener('click', () => {
        // Reset state
        selectedFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        confidenceBar.style.width = '0%';
        
        // Reset UI
        resultSection.classList.add('hidden');
        uploadArea.classList.remove('hidden');
    });
});
