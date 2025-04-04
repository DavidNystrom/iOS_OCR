import UIKit
import AVFoundation

class ViewController: UIViewController, AVCapturePhotoCaptureDelegate {

    // MARK: - Camera Properties
    var captureSession: AVCaptureSession!
    var previewLayer: AVCaptureVideoPreviewLayer!
    var photoOutput: AVCapturePhotoOutput!
    
    // Image view to display the captured image and inference results.
    var resultImageView: UIImageView!
    
    // Reset button to return to camera mode.
    var resetButton: UIButton!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupCamera()
        setupUI()
        TrOCRInference.initializeModels()
    }
    
    // MARK: - Camera & UI Setup
    
    func setupCamera() {
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .photo
        
        guard let videoDevice = AVCaptureDevice.default(for: .video),
              let videoInput = try? AVCaptureDeviceInput(device: videoDevice)
        else {
            print("Unable to access the camera")
            return
        }
        
        if captureSession.canAddInput(videoInput) {
            captureSession.addInput(videoInput)
        }
        
        photoOutput = AVCapturePhotoOutput()
        if captureSession.canAddOutput(photoOutput) {
            captureSession.addOutput(photoOutput)
        }
        
        // Set up the preview layer.
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.frame = view.bounds
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        
        DispatchQueue.global(qos: .background).async {
            self.captureSession.startRunning()
        }
    }
    
    func setupUI() {
        // Capture Button.
        let captureButton = UIButton(frame: CGRect(x: (view.frame.width - 70) / 2,
                                                     y: view.frame.height - 100,
                                                     width: 70,
                                                     height: 70))
        captureButton.backgroundColor = .white
        captureButton.layer.cornerRadius = 35
        captureButton.addTarget(self, action: #selector(capturePhoto), for: .touchUpInside)
        view.addSubview(captureButton)
        
        // Reset Button (top-left).
        resetButton = UIButton(frame: CGRect(x: 20, y: 40, width: 80, height: 40))
        resetButton.backgroundColor = .lightGray
        resetButton.setTitle("Reset", for: .normal)
        resetButton.layer.cornerRadius = 8
        resetButton.addTarget(self, action: #selector(resetTapped), for: .touchUpInside)
        resetButton.isHidden = true
        view.addSubview(resetButton)
        
        // Image view for displaying the captured image.
        resultImageView = UIImageView(frame: view.bounds)
        resultImageView.contentMode = .scaleAspectFit
        resultImageView.isHidden = true
        view.addSubview(resultImageView)
    }
    
    // MARK: - Photo Capture
    
    @objc func capturePhoto() {
        let settings = AVCapturePhotoSettings()
        photoOutput.capturePhoto(with: settings, delegate: self)
    }
    
    func photoOutput(_ output: AVCapturePhotoOutput,
                     didFinishProcessingPhoto photo: AVCapturePhoto,
                     error: Error?) {
        if let error = error {
            print("Error capturing photo: \(error.localizedDescription)")
            return
        }
        
        guard let imageData = photo.fileDataRepresentation(),
              let capturedImage = UIImage(data: imageData)
        else {
            print("Could not convert photo data into an image")
            return
        }
        
        // Stop the camera preview and show the captured image.
        captureSession.stopRunning()
        previewLayer.isHidden = true
        
        // Initially show the original captured image.
        resultImageView.image = capturedImage
        resultImageView.isHidden = false
        
        // Run inference on the captured image.
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                if let detections = try OnnxInference.runInference(on: capturedImage) {
                    // Draw bounding boxes on the ORIGINAL image.
                    let outputImage = self.drawBoundingBoxes(on: capturedImage, detections: detections)
                    DispatchQueue.main.async {
                        self.resultImageView.image = outputImage
                        self.resetButton.isHidden = false
                    }
                } else {
                    print("Inference did not return any results")
                }
            } catch {
                print("Error during inference: \(error)")
            }
        }
    }
    
    // MARK: - Drawing Bounding Boxes
    
    /// Draws bounding boxes (normalized) on the given image.
    func drawBoundingBoxes(on image: UIImage, detections: [DetectionResult]) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(image.size, false, image.scale)
        image.draw(at: .zero)
        
        guard let context = UIGraphicsGetCurrentContext() else {
            return image
        }
        
        context.setLineWidth(2.0)
        context.setStrokeColor(UIColor.red.cgColor)
        
        for detection in detections {
            // Since bounding boxes are normalized, scale them by the original image size.
            let x = detection.boundingBox.origin.x * image.size.width
            let y = detection.boundingBox.origin.y * image.size.height
            let width = detection.boundingBox.size.width * image.size.width
            let height = detection.boundingBox.size.height * image.size.height
            let rect = CGRect(x: x, y: y, width: width, height: height)
            
            context.stroke(rect)
            
            // Optionally draw a label.
            let labelText = "Label: \(detection.label) \(Int(detection.score * 100))%"
            let attributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.systemFont(ofSize: 14),
                .foregroundColor: UIColor.red
            ]
            labelText.draw(at: CGPoint(x: x, y: y), withAttributes: attributes)
        }
        
        let outputImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return outputImage ?? image
    }
    
    // MARK: - Reset Action
    
    @objc func resetTapped() {
        resultImageView.isHidden = true
        resetButton.isHidden = true
        previewLayer.isHidden = false
        
        DispatchQueue.global(qos: .background).async {
            self.captureSession.startRunning()
        }
    }
}
