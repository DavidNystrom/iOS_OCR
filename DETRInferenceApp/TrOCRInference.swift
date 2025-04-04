import UIKit
import onnxruntime_objc

// A helper struct for beam search candidates.
struct BeamCandidate {
    let tokens: [Int]
    let score: Float
}

// This class encapsulates the OCR inference using TrOCR Encoder and Decoder models.
class TrOCRInference {
    
    // ONNX sessions for the encoder and decoder.
    static var encoderSession: ORTSession?
    static var decoderSession: ORTSession?
    
    // Vocabulary mapping: token id -> word.
    static var vocabulary: [Int: String] = [:]
    
    // Beam search parameters.
    static let beamWidth = 5
    static let maxSequenceLength = 50
    
    // Special token IDs (adjust these based on your vocabulary).
    static let bosTokenID = 0   // beginning-of-sequence token id
    static let eosTokenID = 1   // end-of-sequence token id

    // Call this method once (e.g., in AppDelegate or viewDidLoad) to initialize the models and load the vocabulary.
    static func initializeModels() {
        // Load the encoder model.
        guard let encoderModelPath = Bundle.main.path(forResource: "TrOCR_Encoder", ofType: "ort") else {
            print("Failed to find TrOCR Encoder model")
            return
        }
        do {
            let env = try ORTEnv(loggingLevel: .warning)
            encoderSession = try ORTSession(env: env,
                                            modelPath: encoderModelPath,
                                            sessionOptions: ORTSessionOptions())
        } catch {
            print("Error initializing encoder model: \(error)")
        }
        
        // Load the decoder model.
        guard let decoderModelPath = Bundle.main.path(forResource: "TrOCR_Decoder", ofType: "ort") else {
            print("Failed to find TrOCR Decoder model")
            return
        }
        do {
            let env = try ORTEnv(loggingLevel: .warning)
            decoderSession = try ORTSession(env: env,
                                            modelPath: decoderModelPath,
                                            sessionOptions: ORTSessionOptions())
        } catch {
            print("Error initializing decoder model: \(error)")
        }
        
        // Load vocabulary from vocabulary.json.
        loadVocabulary()
    }
    
    // Load the vocabulary JSON file from the bundle.
    static func loadVocabulary() {
        guard let vocabURL = Bundle.main.url(forResource: "vocabulary", withExtension: "json"),
              let data = try? Data(contentsOf: vocabURL),
              let json = try? JSONSerialization.jsonObject(with: data, options: []),
              let vocabDict = json as? [String: Any]
        else {
            print("Failed to load vocabulary.json")
            return
        }
        
        // Assume the JSON maps string token IDs to words.
        for (key, value) in vocabDict {
            if let tokenID = Int(key), let tokenString = value as? String {
                vocabulary[tokenID] = tokenString
            }
        }
    }
    
    // Public method to run OCR on a cropped bounding box image.
    // Returns the recognized text.
    static func runOCR(on image: UIImage) -> String? {
        // 1. Preprocess the cropped image.
        // Here, we define a target size for the OCR model (adjust as needed).
        let targetSize = CGSize(width: 384, height: 384)
        guard let preprocessed = preprocessOCRImage(image, targetSize: targetSize) else {
            print("Failed to preprocess OCR image")
            return nil
        }
        let tensorData = preprocessed.tensorData
        let resizedImage = preprocessed.resizedImage
        
        // Prepare tensor shape [1, 3, H, W] based on the resized image.
        guard let cgImage = resizedImage.cgImage else {
            print("Failed to get CGImage from resized image")
            return nil
        }
        let width = cgImage.width
        let height = cgImage.height
        let shape: [NSNumber] = [1, 3, NSNumber(value: height), NSNumber(value: width)]
        
        // Create input tensor for the encoder.
        let encoderInput: ORTValue
        do {
            encoderInput = try ORTValue(tensorData: tensorData,
                                        elementType: .float,
                                        shape: shape)
        } catch {
            print("Error creating encoder input tensor: \(error)")
            return nil
        }
        
        // 2. Run the encoder model.
        guard let encoderSession = encoderSession else {
            print("Encoder session is not initialized")
            return nil
        }
        var encoderOutputs: [String: ORTValue]
        do {
            // Assuming the input name is "pixel_values" – adjust if needed.
            encoderOutputs = try encoderSession.run(withInputs: ["pixel_values": encoderInput],
                                                    outputNames: ["encoder_hidden_states"],
                                                    runOptions: nil)
        } catch {
            print("Error during encoder inference: \(error)")
            return nil
        }
        
        guard let encoderOutput = encoderOutputs["encoder_hidden_states"] else {
            print("Failed to get encoder output")
            return nil
        }
        
        // 3. Run decoder with beam search.
        guard let bestTokenSequence = decode(encoderOutput: encoderOutput) else {
            print("Decoding failed")
            return nil
        }
        
        // 4. Convert token ids to string.
        let recognizedText = bestTokenSequence.compactMap { vocabulary[$0] }.joined(separator: " ")
        return recognizedText
    }
    
    // Preprocess the image for OCR: resize, normalize, and convert to tensor data.
    static func preprocessOCRImage(_ image: UIImage, targetSize: CGSize) -> (tensorData: NSMutableData, resizedImage: UIImage)? {
        // Resize the image.
        UIGraphicsBeginImageContextWithOptions(targetSize, false, 1.0)
        image.draw(in: CGRect(origin: .zero, size: targetSize))
        guard let resizedImage = UIGraphicsGetImageFromCurrentImageContext() else {
            UIGraphicsEndImageContext()
            return nil
        }
        UIGraphicsEndImageContext()
        
        // Convert the resized image to pixel data.
        guard let cgImage = resizedImage.cgImage else { return nil }
        let width = cgImage.width
        let height = cgImage.height
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
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Create mutable data for tensor.
        let tensorData = NSMutableData()
        let imageSize = width * height
        var floatArray = [Float](repeating: 0, count: imageSize * 3)
        
        // Example normalization values – adjust based on your training.
        let mean: (Float, Float, Float) = (0.5, 0.5, 0.5)
        let std:  (Float, Float, Float) = (0.5, 0.5, 0.5)
        
        for i in 0..<imageSize {
            let offset = i * bytesPerPixel
            let r = Float(pixelData[offset]) / 255.0
            let g = Float(pixelData[offset + 1]) / 255.0
            let b = Float(pixelData[offset + 2]) / 255.0
            
            let normalizedR = (r - mean.0) / std.0
            let normalizedG = (g - mean.1) / std.1
            let normalizedB = (b - mean.2) / std.2
            
            // Channel-first order.
            floatArray[i] = normalizedR
            floatArray[i + imageSize] = normalizedG
            floatArray[i + 2 * imageSize] = normalizedB
        }
        
        let data = Data(bytes: &floatArray, count: floatArray.count * MemoryLayout<Float>.size)
        tensorData.append(data)
        return (tensorData, resizedImage)
    }
    
    // Implements beam search decoding using the decoder model.
    // Returns the best token sequence as an array of token IDs.
    static func decode(encoderOutput: ORTValue) -> [Int]? {
        guard let decoderSession = decoderSession else {
            print("Decoder session is not initialized")
            return nil
        }
        
        // Initialize beam search with a single candidate starting with BOS.
        var beam: [BeamCandidate] = [BeamCandidate(tokens: [bosTokenID], score: 0)]
        
        for _ in 0..<maxSequenceLength {
            var newBeam: [BeamCandidate] = []
            
            // Expand each candidate.
            for candidate in beam {
                // If candidate already ended with EOS, carry it forward.
                if candidate.tokens.last == eosTokenID {
                    newBeam.append(candidate)
                    continue
                }
                
                // Prepare decoder input: candidate.tokens
                let currentSeq = candidate.tokens
                // Convert currentSeq to tensor data.
                var seqFloats = currentSeq.map { Float($0) }
                let seqData = Data(bytes: &seqFloats, count: seqFloats.count * MemoryLayout<Float>.size)
                let seqTensor: ORTValue
                do {
                    // Assume decoder input shape is [1, seq_length].
                    seqTensor = try ORTValue(tensorData: NSMutableData(data: seqData),
                                             elementType: .float,
                                             shape: [1, NSNumber(value: currentSeq.count)])
                } catch {
                    print("Error creating decoder input tensor: \(error)")
                    continue
                }
                
                // Run the decoder model.
                // Assuming the decoder takes two inputs: "decoder_input_ids" and "encoder_outputs".
                var decoderInputs: [String: ORTValue] = [
                    "decoder_input_ids": seqTensor,
                    "encoder_outputs": encoderOutput
                ]
                let outputs: [String: ORTValue]
                do {
                    // Assuming the output tensor is named "logits".
                    outputs = try decoderSession.run(withInputs: decoderInputs,
                                                     outputNames: ["logits"],
                                                     runOptions: nil)
                } catch {
                    print("Error during decoder inference: \(error)")
                    continue
                }
                
                guard let logitsTensor = outputs["logits"],
                      let logitsData = try? logitsTensor.tensorData(),
                      let logitsArray: [Float] = arrayCopiedFromData(logitsData as Data)
                else {
                    print("Failed to obtain decoder logits")
                    continue
                }
                
                // Assume logits shape is [1, vocab_size]. Extract probabilities.
                // Compute softmax.
                let vocabSize = logitsArray.count
                let expValues = logitsArray.map { exp($0) }
                let sumExp = expValues.reduce(0, +)
                let probabilities = expValues.map { $0 / sumExp }
                
                // Get top beamWidth candidates for the next token.
                let topIndices = probabilities.enumerated()
                    .sorted(by: { $0.element > $1.element })
                    .prefix(beamWidth)
                    .map { ($0.offset, probabilities[$0.offset]) }
                
                // Expand the candidate for each top token.
                for (tokenID, prob) in topIndices {
                    let newTokens = candidate.tokens + [tokenID]
                    // Use log probability to accumulate score.
                    let newScore = candidate.score + log(prob)
                    newBeam.append(BeamCandidate(tokens: newTokens, score: newScore))
                }
            }
            
            // Keep only the best beamWidth candidates.
            beam = newBeam.sorted(by: { $0.score > $1.score }).prefix(beamWidth).map { $0 }
            
            // If all candidates have ended with EOS, stop early.
            if beam.allSatisfy({ $0.tokens.last == eosTokenID }) {
                break
            }
        }
        
        // Return the candidate with the best score.
        return beam.sorted(by: { $0.score > $1.score }).first?.tokens
    }
    
    // Helper to convert Data to array of a specified type.
    private static func arrayCopiedFromData<T>(_ data: Data) -> [T]? {
        guard data.count % MemoryLayout<T>.stride == 0 else { return nil }
        return data.withUnsafeBytes { bytes -> [T] in
            return Array(bytes.bindMemory(to: T.self))
        }
    }
}
