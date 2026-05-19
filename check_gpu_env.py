import sys
import onnxruntime as ort
import insightface
from insightface.app import FaceAnalysis

def check_gpu():
    print("--- Environment Check (CoreML/CPU) ---")
    print(f"Python: {sys.version.split(' ')[0]}")
    
    # Check ONNX Runtime Providers
    providers = ort.get_available_providers()
    print(f"Available ONNX Providers: {providers}")
    
    if 'CoreMLExecutionProvider' not in providers:
        print("\n[WARNING] CoreMLExecutionProvider NOT found. It will fall back to CPU.")
        print("For M-series Macs, CoreML provides hardware acceleration.")
    else:
        print("\n[SUCCESS] CoreML (Apple Silicon) is available!")
    
    # Check InsightFace Initialization
    print("\n--- Initializing InsightFace (CoreML/CPU) ---")
    try:
        available_providers = ort.get_available_providers()
        providers = []
        if 'CUDAExecutionProvider' in available_providers: providers.append('CUDAExecutionProvider')
        elif 'CoreMLExecutionProvider' in available_providers: providers.append('CoreMLExecutionProvider')
        elif 'DmlExecutionProvider' in available_providers: providers.append('DmlExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        app = FaceAnalysis(name='buffalo_l', providers=providers)
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("[SUCCESS] InsightFace initialized with CoreML/CPU!")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to initialize InsightFace: {e}")
        return False

if __name__ == "__main__":
    check_gpu()
