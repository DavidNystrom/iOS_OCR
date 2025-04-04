import UIKit
import onnxruntime_objc

// Structure to hold detection results.
struct DetectionResult {
    let boundingBox: CGRect  // Normalized [0,1] coordinates
    let label: Int           // Label index (you may map this to a class name)
    let score: Float         // Confidence score
}

class OnnxInference {
    
    /// Runs inference on a given UIImage and returns an array of detection results.
    /// The image is preprocessed (resized so that the shortest side is 800) for inference.
    static func runInference(on image: UIImage) throws -> [DetectionResult]? {
        // 1. Initialize ONNX Runtime environment and session.
        var ortEnvironment: ORTEnv?
        var ortSession: ORTSession?
        do {
            ortEnvironment = try ORTEnv(loggingLevel: .warning)
        } catch {
            print("Failed to initialize ONNX Runtime Environment: \(error)")
            return nil
        }
        
        guard let modelPath = Bundle.main.path(forResource: "detr", ofType: "ort") else {
            print("Failed to find the model file.")
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
        
        // 2. Preprocess the image into a tensor.
        //    We get a tuple: (tensorData, resizedImage)
        guard let preprocessed = preprocess(image: image) else {
            print("Failed to preprocess image")
            return nil
        }
        let inputTensorData = preprocessed.tensorData
        let resizedImage = preprocessed.resizedImage
        
        // Get the resized image dimensions for constructing the tensor shape.
        guard let resizedCG = resizedImage.cgImage else {
            print("Failed to get resized image CGImage")
            return nil
        }
        let newWidth = resizedCG.width
        let newHeight = resizedCG.height
        
        // Use the actual resized dimensions.
        let shape: [NSNumber] = [1, 3, NSNumber(value: newHeight), NSNumber(value: newWidth)]
        
        let inputTensor: ORTValue
        do {
            inputTensor = try ORTValue(tensorData: inputTensorData,
                                       elementType: .float,
                                       shape: shape)
        } catch {
            print("Error creating input tensor: \(error)")
            return nil
        }
        
        // 3. Run inference using the correct output names.
        let outputNames: Set<String> = ["pred_logits", "pred_boxes"]
        let outputTensor: [String: ORTValue]
        do {
            outputTensor = try ortSession?.run(withInputs: ["input": inputTensor],
                                               outputNames: outputNames,
                                               runOptions: nil) ?? [:]
        } catch {
            print("Error during inference: \(error)")
            return nil
        }
        
        // 4. Extract data for "pred_logits" and "pred_boxes".
        guard
            let predLogitsData = try outputTensor["pred_logits"]?.tensorData(),
            let predBoxesData  = try outputTensor["pred_boxes"]?.tensorData()
        else {
            print("Failed to retrieve output tensors")
            return nil
        }
        
        guard
            let predLogitsArray: [Float] = arrayCopiedFromData(predLogitsData as Data),
            let predBoxesArray:  [Float] = arrayCopiedFromData(predBoxesData as Data)
        else {
            print("Failed to parse output tensor data")
            return nil
        }
        
        // 5. Post-process outputs to get detection results.
        // Adjust these constants for your model:
        let batchSize = 1
        let numQueries = 100
        let numClasses = 3  // For example: two object classes + background.
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
            
            // Find the best class and its score.
            var bestClass = -1
            var bestScore: Float = -1.0
            for c in 0..<numClasses {
                let score = probabilities[c]
                if score > bestScore {
                    bestScore = score
                    bestClass = c
                }
            }
            
            // Skip background or low-confidence detections.
            if bestClass == backgroundClassIndex || bestScore < scoreThreshold {
                continue
            }
            
            // Get bounding box (cx, cy, w, h) for query i.
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
        
        return results
    }
    
    /// Preprocesses a UIImage by resizing (keeping the aspect ratio so that the shortest side is targetShortSide),
    /// normalizing, and converting it to a mutable data buffer in channel-first order.
    /// Returns a tuple containing the tensor data and the resized image.
    static func preprocess(image: UIImage, targetShortSide: CGFloat = 800) -> (tensorData: NSMutableData, resizedImage: UIImage)? {
        // Get original dimensions.
        guard let cgImage = image.cgImage else { return nil }
        let originalWidth = CGFloat(cgImage.width)
        let originalHeight = CGFloat(cgImage.height)
        
        // Compute scale factor so that the shortest side equals targetShortSide.
        let scale = targetShortSide / min(originalWidth, originalHeight)
        let newWidth = Int(originalWidth * scale)
        let newHeight = Int(originalHeight * scale)
        
        // Resize the image.
        UIGraphicsBeginImageContextWithOptions(CGSize(width: newWidth, height: newHeight), false, 1.0)
        image.draw(in: CGRect(x: 0, y: 0, width: newWidth, height: newHeight))
        guard let resizedImage = UIGraphicsGetImageFromCurrentImageContext() else {
            UIGraphicsEndImageContext()
            return nil
        }
        UIGraphicsEndImageContext()
        
        // Convert the resized image to pixel data.
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
        
        // Create an NSMutableData object to hold the tensor data.
        let tensorData = NSMutableData()
        let imageSize = width * height
        var floatArray = [Float](repeating: 0, count: imageSize * 3)
        
        // Standard ImageNet normalization values.
        let mean: (Float, Float, Float) = (0.485, 0.456, 0.406)
        let std:  (Float, Float, Float) = (0.229, 0.224, 0.225)
        
        // Convert pixel values to floats in channel-first order and normalize.
        for i in 0..<imageSize {
            let offset = i * bytesPerPixel
            // Scale pixel values to [0,1]
            let r = Float(pixelData[offset]) / 255.0
            let g = Float(pixelData[offset + 1]) / 255.0
            let b = Float(pixelData[offset + 2]) / 255.0
            
            // Normalize using ImageNet stats.
            let normalizedR = (r - mean.0) / std.0
            let normalizedG = (g - mean.1) / std.1
            let normalizedB = (b - mean.2) / std.2
            
            floatArray[i] = normalizedR
            floatArray[i + imageSize] = normalizedG
            floatArray[i + 2 * imageSize] = normalizedB
        }
        
        // Convert the floatArray to Data and append to tensorData.
        let data = Data(bytes: &floatArray, count: floatArray.count * MemoryLayout<Float>.size)
        tensorData.append(data)
        
        return (tensorData, resizedImage)
    }

    
    /// Helper method to convert Data to an array of a specified type.
    private static func arrayCopiedFromData<T>(_ data: Data) -> [T]? {
        guard data.count % MemoryLayout<T>.stride == 0 else { return nil }
        return data.withUnsafeBytes { bytes -> [T] in
            return Array(bytes.bindMemory(to: T.self))
        }
    }
}
