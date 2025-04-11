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
    
    // Text view to display recognized text separately.
    var recognizedTextView: UITextView!
    
    // Inference Time Label (displayed at the top-right).
    var inferenceTimeLabel: UILabel!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupCamera()
        setupUI()
        // Initialize the OCR models.
        TrOCRInference.initializeModels()
    }
    
    // MARK: - Camera & UI Setup
    
    func setupCamera() {
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .photo
        
        guard let videoDevice = AVCaptureDevice.default(for: .video),
              let videoInput = try? AVCaptureDeviceInput(device: videoDevice) else {
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
        
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.frame = view.bounds
        previewLayer.videoGravity = .resizeAspect
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
        
        // Inference Time Label (top-right).
        inferenceTimeLabel = UILabel(frame: CGRect(x: view.frame.width - 160, y: 40, width: 150, height: 40))
        inferenceTimeLabel.backgroundColor = UIColor.black.withAlphaComponent(0.5)
        inferenceTimeLabel.textColor = .white
        inferenceTimeLabel.font = UIFont.systemFont(ofSize: 14)
        inferenceTimeLabel.textAlignment = .center
        inferenceTimeLabel.text = "Inference: -- s"
        view.addSubview(inferenceTimeLabel)
        
        // Image view for displaying the captured image.
        resultImageView = UIImageView(frame: view.bounds)
        resultImageView.contentMode = .scaleAspectFit
        resultImageView.isHidden = true
        view.addSubview(resultImageView)
        
        // Recognized text view to display OCR results.
        recognizedTextView = UITextView(frame: CGRect(x: 16,
                                                        y: view.frame.height - 220,
                                                        width: view.frame.width - 32,
                                                        height: 200))
        recognizedTextView.backgroundColor = UIColor.black.withAlphaComponent(0.7)
        recognizedTextView.textColor = UIColor.white
        recognizedTextView.font = UIFont.systemFont(ofSize: 18)
        recognizedTextView.isEditable = false
        recognizedTextView.isHidden = true
        recognizedTextView.layer.cornerRadius = 8
        recognizedTextView.clipsToBounds = true
        view.addSubview(recognizedTextView)
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
              let capturedImage = UIImage(data: imageData) else {
            print("Could not convert photo data into an image")
            return
        }
        
        print("Captured image size: \(capturedImage.size)")
        
        captureSession.stopRunning()
        previewLayer.isHidden = true
        
        DispatchQueue.main.async {
            self.resultImageView.image = capturedImage
            self.resultImageView.isHidden = false
        }
        
        // Capture overall inference start time.
        let inferenceStart = Date()
        
        // Run detection and OCR inference.
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                if let inferenceResult = try DETRInference.runInference(on: capturedImage) {
                    let detections = inferenceResult.detections
                    let resizedImage = inferenceResult.resizedImage
                    
                    let outputImage = self.drawBoundingBoxes(on: resizedImage, detections: detections)
                    DispatchQueue.main.async {
                        self.resultImageView.image = outputImage
                        self.resetButton.isHidden = false
                    }
                    
                    // Sort detections for OCR processing.
                    let sortedDetections = detections.sorted { (d1, d2) -> Bool in
                        let yDifference = abs(d1.boundingBox.origin.y - d2.boundingBox.origin.y)
                        if yDifference < 0.05 {
                            return d1.boundingBox.origin.x < d2.boundingBox.origin.x
                        } else {
                            return d1.boundingBox.origin.y < d2.boundingBox.origin.y
                        }
                    }
                    
                    var recognizedTexts = [String]()
                    for detection in sortedDetections {
                        let x = detection.boundingBox.origin.x * resizedImage.size.width
                        let y = detection.boundingBox.origin.y * resizedImage.size.height
                        let width = detection.boundingBox.size.width * resizedImage.size.width
                        let height = detection.boundingBox.size.height * resizedImage.size.height
                        let rect = CGRect(x: x, y: y, width: width, height: height)
                        if let croppedImage = self.crop(image: resizedImage, with: rect) {
                            if let recognizedText = TrOCRInference.runOCR(on: croppedImage) {
                                recognizedTexts.append("Label: \(detection.label) \(Int(detection.score * 100))% – \(recognizedText)")
                            }
                        }
                    }
                    
                    DispatchQueue.main.async {
                        self.recognizedTextView.text = recognizedTexts.joined(separator: "\n")
                        self.recognizedTextView.isHidden = false
                        
                        let totalInferenceTime = Date().timeIntervalSince(inferenceStart)
                        self.inferenceTimeLabel.text = String(format: "Inference: %.2f s", totalInferenceTime)
                    }
                } else {
                    print("Inference did not return any results")
                }
            } catch {
                print("Error during inference: \(error)")
            }
        }
    }
    
    // MARK: - Drawing Bounding Boxes (without overlay text)
    
    func drawBoundingBoxes(on image: UIImage, detections: [DetectionResult]) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(image.size, false, image.scale)
        image.draw(at: .zero)
        
        guard let context = UIGraphicsGetCurrentContext() else {
            return image
        }
        
        context.setLineWidth(2.0)
        context.setStrokeColor(UIColor.red.cgColor)
        
        let limitedDetections = detections.sorted { $0.score > $1.score }.prefix(10)
        
        for detection in limitedDetections {
            let x = detection.boundingBox.origin.x * image.size.width
            let y = detection.boundingBox.origin.y * image.size.height
            let width = detection.boundingBox.size.width * image.size.width
            let height = detection.boundingBox.size.height * image.size.height
            let rect = CGRect(x: x, y: y, width: width, height: height)
            context.stroke(rect)
        }
        
        let outputImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return outputImage ?? image
    }
    
    func crop(image: UIImage, with rect: CGRect) -> UIImage? {
        guard let cgImage = image.cgImage else { return nil }
        let scaledRect = CGRect(x: rect.origin.x * image.scale,
                                y: rect.origin.y * image.scale,
                                width: rect.size.width * image.scale,
                                height: rect.size.height * image.scale)
        if let croppedCGImage = cgImage.cropping(to: scaledRect) {
            return UIImage(cgImage: croppedCGImage, scale: image.scale, orientation: image.imageOrientation)
        }
        return nil
    }
    
    // MARK: - Reset Action
    
    @objc func resetTapped() {
        resultImageView.isHidden = true
        recognizedTextView.isHidden = true
        resetButton.isHidden = true
        previewLayer.isHidden = false
        
        DispatchQueue.global(qos: .background).async {
            self.captureSession.startRunning()
        }
    }
}
