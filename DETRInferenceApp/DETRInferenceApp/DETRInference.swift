import UIKit
import onnxruntime_objc

// MARK: - UIImage Orientation Normalization Extension
extension UIImage {
    /// Returns an image that is oriented upright, discarding the orientation metadata.
    var normalizedImage: UIImage {
        if imageOrientation == .up {
            return self
        }
        UIGraphicsBeginImageContextWithOptions(size, false, scale)
        self.draw(in: CGRect(origin: .zero, size: size))
        let normalized = UIGraphicsGetImageFromCurrentImageContext() ?? self
        UIGraphicsEndImageContext()
        return normalized
    }
}

// MARK: - DetectionResult
struct DetectionResult {
    let boundingBox: CGRect  // Normalized coordinates [0,1]
    let label: Int           // Label index (you may map this to a class name)
    let score: Float         // Confidence score
}

// MARK: - DETRInference Class
class DETRInference {
    
    /// Runs inference on a given UIImage and returns detection results and the resized image.
    static func runInference(on image: UIImage) throws -> (detections: [DetectionResult], resizedImage: UIImage)? {
        let overallStart = Date()
        
        // Step 1: Initialize ONNX Runtime environment and session.
        let initStart = Date()
        var ortEnvironment: ORTEnv?
        var ortSession: ORTSession?
        do {
            ortEnvironment = try ORTEnv(loggingLevel: .warning)
        } catch {
            print("Failed to initialize ONNX Runtime Environment: \(error)")
            return nil
        }
        
        guard let modelPath = Bundle.main.path(forResource: "detr", ofType: "ort") else {
            print("Failed to find the DETR model file.")
            return nil
        }
        
        do {
            ortSession = try ORTSession(env: ortEnvironment!,
                                        modelPath: modelPath,
                                        sessionOptions: ORTSessionOptions())
        } catch {
            print("Failed to create ONNX Runtime Session: \(error)")
            return nil
        }
        print("Step 1 (Initialization) took: \(Date().timeIntervalSince(initStart)) seconds")
        
        // Step 2: Preprocess the image into a tensor.
        let preprocessStart = Date()
        let normalizedImage = image.normalizedImage
        
        guard let preprocessed = preprocess(image: normalizedImage) else {
            print("Failed to preprocess image")
            return nil
        }
        print("Step 2 (Preprocessing) took: \(Date().timeIntervalSince(preprocessStart)) seconds")
        
        let inputTensorData = preprocessed.tensorData
        let resizedImage = preprocessed.resizedImage
        
        guard let resizedCG = resizedImage.cgImage else {
            print("Failed to get resized image CGImage")
            return nil
        }
        let newWidth = resizedCG.width
        let newHeight = resizedCG.height
        let shape: [NSNumber] = [1, 3, NSNumber(value: newHeight), NSNumber(value: newWidth)]
        
        // Step 3: Create input tensor.
        let tensorCreationStart = Date()
        let inputTensor: ORTValue
        do {
            inputTensor = try ORTValue(tensorData: inputTensorData,
                                       elementType: .float,
                                       shape: shape)
        } catch {
            print("Error creating input tensor: \(error)")
            return nil
        }
        print("Step 3 (Tensor Creation) took: \(Date().timeIntervalSince(tensorCreationStart)) seconds")
        
        // Step 4: Run inference using the DETR model.
        let inferenceStart = Date()
        let outputNames: Set<String> = ["pred_logits", "pred_boxes"]
        let outputTensor: [String: ORTValue]
        do {
            outputTensor = try ortSession?.run(withInputs: ["input": inputTensor],
                                               outputNames: outputNames,
                                               runOptions: nil) ?? [:]
        } catch {
            print("Error during DETR inference: \(error)")
            return nil
        }
        print("Step 4 (Model Inference) took: \(Date().timeIntervalSince(inferenceStart)) seconds")
        
        // Step 5: Post-process outputs to get detection results.
        let postProcessStart = Date()
        guard let predLogitsData = try? outputTensor["pred_logits"]?.tensorData(),
              let predBoxesData  = try? outputTensor["pred_boxes"]?.tensorData() else {
            print("Failed to retrieve output tensors")
            return nil
        }
        
        guard let predLogitsArray: [Float] = arrayCopiedFromData(predLogitsData as Data),
              let predBoxesArray:  [Float] = arrayCopiedFromData(predBoxesData as Data) else {
            print("Failed to parse output tensor data")
            return nil
        }
        
        // Constants for the model.
        let batchSize = 1
        let numQueries = 100
        let numClasses = 3  // Example: two object classes + background.
        let backgroundClassIndex = 2
        let scoreThreshold: Float = 0.9
        
        guard predLogitsArray.count == batchSize * numQueries * numClasses,
              predBoxesArray.count == batchSize * numQueries * 4 else {
            print("Output shapes are not as expected. Check your model dimensions.")
            return nil
        }
        
        var results = [DetectionResult]()
        for i in 0..<numQueries {
            // Extract logits for query i.
            let start = i * numClasses
            let end   = start + numClasses
            let logitsSlice = Array(predLogitsArray[start..<end])
            
            // Compute softmax to get probabilities.
            let expScores = logitsSlice.map { exp($0) }
            let sumExp = expScores.reduce(0, +)
            let probabilities = expScores.map { $0 / sumExp }
            
            // Find the best class.
            var bestClass = -1
            var bestScore: Float = -1.0
            for c in 0..<numClasses {
                let score = probabilities[c]
                if score > bestScore {
                    bestScore = score
                    bestClass = c
                }
            }
            
            // Skip background or low confidence detections.
            if bestClass == backgroundClassIndex || bestScore < scoreThreshold {
                continue
            }
            
            // Get bounding box (center format: [cx, cy, w, h]).
            let bx = predBoxesArray[i * 4 + 0]
            let by = predBoxesArray[i * 4 + 1]
            let bw = predBoxesArray[i * 4 + 2]
            let bh = predBoxesArray[i * 4 + 3]
            
            // Convert from center format to normalized [xMin, yMin, xMax, yMax].
            let xMin = bx - bw / 2
            let yMin = by - bh / 2
            let xMax = bx + bw / 2
            let yMax = by + bh / 2
            
            results.append(
                DetectionResult(
                    boundingBox: CGRect(x: CGFloat(xMin),
                                        y: CGFloat(yMin),
                                        width: CGFloat(xMax - xMin),
                                        height: CGFloat(yMax - yMin)),
                    label: bestClass,
                    score: bestScore
                )
            )
        }
        
        print("Step 5 (Post-Processing) took: \(Date().timeIntervalSince(postProcessStart)) seconds")
        print("Total DETR inference time: \(Date().timeIntervalSince(overallStart)) seconds")
        
        return (results, resizedImage)
    }
    
    /// Preprocesses a UIImage by resizing it so that the shortest side becomes targetShortSide,
    /// normalizing the pixel values, and converting it into a mutable data buffer (channel-first order).
    /// This method preserves the original aspect ratio.
    static func preprocess(image: UIImage, targetShortSide: CGFloat = 800) -> (tensorData: NSMutableData, resizedImage: UIImage)? {
        guard let cgImage = image.cgImage else { return nil }
        let originalWidth = CGFloat(cgImage.width)
        let originalHeight = CGFloat(cgImage.height)
        
        let scale = targetShortSide / min(originalWidth, originalHeight)
        let newWidth = Int(originalWidth * scale)
        let newHeight = Int(originalHeight * scale)
        
        UIGraphicsBeginImageContextWithOptions(CGSize(width: newWidth, height: newHeight), false, image.scale)
        image.draw(in: CGRect(x: 0, y: 0, width: newWidth, height: newHeight))
        guard let resizedImage = UIGraphicsGetImageFromCurrentImageContext() else {
            UIGraphicsEndImageContext()
            return nil
        }
        UIGraphicsEndImageContext()
        
        guard let resizedCGImage = resizedImage.cgImage else { return nil }
        let width = resizedCGImage.width
        let height = resizedCGImage.height
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let totalBytes = height * bytesPerRow
        var pixelData = [UInt8](repeating: 0, count: totalBytes)
        
        guard let context = CGContext(data: &pixelData,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: bytesPerRow,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { return nil }
        context.draw(resizedCGImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        let tensorData = NSMutableData()
        let imageSize = width * height
        var floatArray = [Float](repeating: 0, count: imageSize * 3)
        
        // Standard ImageNet normalization values.
        let mean: (Float, Float, Float) = (0.485, 0.456, 0.406)
        let std:  (Float, Float, Float) = (0.229, 0.224, 0.225)
        
        for i in 0..<imageSize {
            let offset = i * bytesPerPixel
            let r = Float(pixelData[offset]) / 255.0
            let g = Float(pixelData[offset + 1]) / 255.0
            let b = Float(pixelData[offset + 2]) / 255.0
            
            let normalizedR = (r - mean.0) / std.0
            let normalizedG = (g - mean.1) / std.1
            let normalizedB = (b - mean.2) / std.2
            
            floatArray[i] = normalizedR
            floatArray[i + imageSize] = normalizedG
            floatArray[i + 2 * imageSize] = normalizedB
        }
        
        let data = Data(bytes: &floatArray, count: floatArray.count * MemoryLayout<Float>.size)
        tensorData.append(data)
        
        return (tensorData, resizedImage)
    }
    
    /// Helper method to convert Data to an array of a specified type.
    private static func arrayCopiedFromData<T>(_ data: Data) -> [T]? {
        guard data.count % MemoryLayout<T>.stride == 0 else { return nil }
        return data.withUnsafeBytes { bytes -> [T] in
            Array(bytes.bindMemory(to: T.self))
        }
    }
}
