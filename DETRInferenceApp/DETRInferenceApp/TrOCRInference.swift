import UIKit
import onnxruntime_objc

// MARK: - BeamCandidate Struct
struct BeamCandidate {
    let tokens: [Int]
    let score: Float
}

// MARK: - TrOCRInference Class
class TrOCRInference {
    
    // ONNX sessions for the encoder and decoder.
    static var encoderSession: ORTSession?
    static var decoderSession: ORTSession?
    
    // Vocabulary mapping: token id -> word.
    static var vocabulary: [Int: String] = [:]
    
    // Beam search parameters.
    static let beamWidth = 1
    static let maxSequenceLength = 8
    
    // Special token IDs.
    static let bosTokenID = 0   // beginning-of-sequence
    static let eosTokenID = 2   // end-of-sequence
    
    /// Initializes the encoder and decoder models and loads the vocabulary.
    static func initializeModels() {
        print("Initializing TrOCR models...")
        
        // Load the encoder model.
        guard let encoderModelPath = Bundle.main.path(forResource: "encoder_model_quantized", ofType: "ort") else {
            print("Failed to find TrOCR Encoder model")
            return
        }
        do {
            let env = try ORTEnv(loggingLevel: .warning)
            encoderSession = try ORTSession(env: env,
                                            modelPath: encoderModelPath,
                                            sessionOptions: ORTSessionOptions())
            print("Encoder model initialized successfully.")
        } catch {
            print("Error initializing encoder model: \(error)")
        }
        
        // Load the decoder model.
        guard let decoderModelPath = Bundle.main.path(forResource: "decoder_model_quantized", ofType: "ort") else {
            print("Failed to find TrOCR Decoder model")
            return
        }
        do {
            let env = try ORTEnv(loggingLevel: .warning)
            decoderSession = try ORTSession(env: env,
                                            modelPath: decoderModelPath,
                                            sessionOptions: ORTSessionOptions())
            print("Decoder model initialized successfully.")
        } catch {
            print("Error initializing decoder model: \(error)")
        }
        
        // Load vocabulary.
        loadVocabulary()
    }
    
    /// Loads the vocabulary from vocab.json.
    static func loadVocabulary() {
        guard let vocabURL = Bundle.main.url(forResource: "vocab", withExtension: "json"),
              let data = try? Data(contentsOf: vocabURL),
              let json = try? JSONSerialization.jsonObject(with: data, options: []) else {
            print("Failed to load vocab.json")
            return
        }
        
        if let tokenToID = json as? [String: Int] {
            for (token, id) in tokenToID {
                vocabulary[id] = token
            }
            print("Vocabulary loaded with \(vocabulary.count) entries.")
        } else {
            print("vocab.json has an unexpected format: \(json)")
        }
    }
    
    /// Post-processes the raw token string to handle subword markers.
    static func postProcess(_ rawText: String) -> String {
        let processed = rawText.replacingOccurrences(of: "Ġ", with: " ").trimmingCharacters(in: .whitespacesAndNewlines)
        return processed
    }
    
    /// Runs OCR on a cropped bounding box image and returns the recognized text.
    static func runOCR(on image: UIImage) -> String? {
        print("Starting OCR inference...")
        let overallStart = Date()
        
        // Step 1: Preprocess the OCR image.
        let targetSize = CGSize(width: 384, height: 384)
        let preprocessStart = Date()
        guard let preprocessed = preprocessOCRImage(image, targetSize: targetSize) else {
            print("Failed to preprocess OCR image")
            return nil
        }
        print("OCR image preprocessing took: \(Date().timeIntervalSince(preprocessStart)) seconds")
        
        let tensorData = preprocessed.tensorData
        let resizedImage = preprocessed.resizedImage
        
        guard let cgImage = resizedImage.cgImage else {
            print("Failed to get CGImage from resized image")
            return nil
        }
        let width = cgImage.width
        let height = cgImage.height
        let shape: [NSNumber] = [1, 3, NSNumber(value: height), NSNumber(value: width)]
        print("Encoder tensor shape: [1, 3, \(height), \(width)]")
        
        let encoderInput: ORTValue
        do {
            encoderInput = try ORTValue(tensorData: tensorData,
                                        elementType: .float,
                                        shape: shape)
            print("Encoder input tensor created successfully.")
        } catch {
            print("Error creating encoder input tensor: \(error)")
            return nil
        }
        
        guard let encoderSession = encoderSession else {
            print("Encoder session is not initialized")
            return nil
        }
        
        // Step 2: Run encoder inference.
        let encoderStart = Date()
        var encoderOutputs: [String: ORTValue]
        do {
            print("Running encoder inference...")
            encoderOutputs = try encoderSession.run(withInputs: ["pixel_values": encoderInput],
                                                    outputNames: ["last_hidden_state"],
                                                    runOptions: nil)
            print("Encoder inference completed, took: \(Date().timeIntervalSince(encoderStart)) seconds")
        } catch {
            print("Error during encoder inference: \(error)")
            return nil
        }
        
        guard let encoderOutput = encoderOutputs["last_hidden_state"] else {
            print("Failed to get encoder output")
            return nil
        }
        
        // Step 3: Run decoder beam search.
        let decoderStart = Date()
        guard let bestTokenSequence = decode(encoderOutput: encoderOutput) else {
            print("Decoding failed")
            return nil
        }
        print("Decoder beam search took: \(Date().timeIntervalSince(decoderStart)) seconds")
        
        // Build recognized text from the best token sequence.
        let tokenStrings = bestTokenSequence.compactMap { vocabulary[$0] }
        let rawText = tokenStrings.joined()
        let recognizedText = postProcess(rawText)
        
        print("Total OCR inference time: \(Date().timeIntervalSince(overallStart)) seconds")
        print("OCR inference completed. Recognized text: \(recognizedText)")
        return recognizedText
    }
    
    /// Preprocesses the OCR image: resize, normalize, and convert to tensor data.
    static func preprocessOCRImage(_ image: UIImage, targetSize: CGSize) -> (tensorData: NSMutableData, resizedImage: UIImage)? {
        print("Preprocessing OCR image...")
        UIGraphicsBeginImageContextWithOptions(targetSize, false, 1.0)
        image.draw(in: CGRect(origin: .zero, size: targetSize))
        guard let resizedImage = UIGraphicsGetImageFromCurrentImageContext() else {
            UIGraphicsEndImageContext()
            print("Image resizing failed.")
            return nil
        }
        UIGraphicsEndImageContext()
        print("Image resized successfully.")
        
        guard let cgImage = resizedImage.cgImage else {
            print("Failed to get CGImage from resized image.")
            return nil
        }
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
        else {
            print("Failed to create CGContext for image.")
            return nil
        }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        print("Extracted pixel data from image.")
        
        let tensorData = NSMutableData()
        let imageSize = width * height
        var floatArray = [Float](repeating: 0, count: imageSize * 3)
        
        // Example normalization values.
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
            
            floatArray[i] = normalizedR
            floatArray[i + imageSize] = normalizedG
            floatArray[i + 2 * imageSize] = normalizedB
        }
        
        let data = Data(bytes: &floatArray, count: floatArray.count * MemoryLayout<Float>.size)
        tensorData.append(data)
        print("Image preprocessed into tensor data with \(data.count) bytes.")
        return (tensorData, resizedImage)
    }
    
    /// Implements beam search decoding using the decoder model.
    static func decode(encoderOutput: ORTValue) -> [Int]? {
        print("Starting decoder beam search...")
        guard let decoderSession = decoderSession else {
            print("Decoder session is not initialized")
            return nil
        }
        
        var beam: [BeamCandidate] = [BeamCandidate(tokens: [bosTokenID], score: 0)]
        
        for step in 0..<maxSequenceLength {
            print("Beam search iteration \(step)")
            var newBeam: [BeamCandidate] = []
            
            for candidate in beam {
                if candidate.tokens.last == eosTokenID {
                    print("Candidate already ended with EOS: \(candidate.tokens)")
                    newBeam.append(candidate)
                    continue
                }
                
                let currentSeq = candidate.tokens
                print("Decoding candidate: \(currentSeq)")
                var seqInts = currentSeq.map { Int64($0) }
                let seqData = Data(bytes: &seqInts, count: seqInts.count * MemoryLayout<Int64>.size)
                let seqTensor: ORTValue
                do {
                    seqTensor = try ORTValue(tensorData: NSMutableData(data: seqData),
                                             elementType: .int64,
                                             shape: [1, NSNumber(value: currentSeq.count)])
                    print("Decoder input tensor created for sequence: \(currentSeq)")
                } catch {
                    print("Error creating decoder input tensor: \(error)")
                    continue
                }
                
                let decoderInputs: [String: ORTValue] = [
                    "input_ids": seqTensor,
                    "encoder_hidden_states": encoderOutput
                ]
                let outputs: [String: ORTValue]
                do {
                    print("Running decoder inference for sequence: \(currentSeq)")
                    outputs = try decoderSession.run(withInputs: decoderInputs,
                                                     outputNames: ["logits"],
                                                     runOptions: nil)
                    print("Decoder inference completed for sequence: \(currentSeq)")
                } catch {
                    print("Error during decoder inference: \(error)")
                    continue
                }
                
                guard let logitsTensor = outputs["logits"] else {
                    print("Logits output not found for sequence: \(currentSeq)")
                    continue
                }
                
                do {
                    let logitsData = try logitsTensor.tensorData()
                    print("Logits data size: \(logitsData.count) bytes")
                    
                    if let typeAndShapeInfo = try? logitsTensor.tensorTypeAndShapeInfo() {
                        print("Logits tensor shape: \(typeAndShapeInfo.shape)")
                    } else {
                        print("Unable to retrieve logits tensor shape.")
                    }
                    
                    guard let fullLogits: [Float] = arrayCopiedFromData(logitsData as Data) else {
                        print("Failed to convert logits data to [Float]")
                        continue
                    }
                    
                    let currentSeqLength = currentSeq.count
                    let totalElements = fullLogits.count
                    guard totalElements % currentSeqLength == 0 else {
                        print("Mismatch: total elements \(totalElements) not divisible by sequence length \(currentSeqLength)")
                        continue
                    }
                    let vocabSize = totalElements / currentSeqLength
                    print("Determined vocab size: \(vocabSize)")
                    
                    let startIndex = (currentSeqLength - 1) * vocabSize
                    let endIndex = startIndex + vocabSize
                    if endIndex > fullLogits.count {
                        print("Error: Calculated end index \(endIndex) exceeds logits count \(fullLogits.count)")
                        continue
                    }
                    let lastTokenLogits = Array(fullLogits[startIndex..<endIndex])
                    
                    let expValues = lastTokenLogits.map { exp($0) }
                    let sumExp = expValues.reduce(0, +)
                    let probabilities = expValues.map { $0 / sumExp }
                    
                    let topIndices = probabilities.enumerated()
                        .sorted(by: { $0.element > $1.element })
                        .prefix(beamWidth)
                        .map { ($0.offset, probabilities[$0.offset]) }
                    
                    for (tokenID, prob) in topIndices {
                        let newTokens = candidate.tokens + [tokenID]
                        let newScore = candidate.score + log(prob)
                        let tokenStr = vocabulary[tokenID] ?? "\(tokenID)"
                        print("Expanding candidate \(candidate.tokens) with token \(tokenStr) (ID: \(tokenID)) (new score: \(newScore))")
                        newBeam.append(BeamCandidate(tokens: newTokens, score: newScore))
                    }
                    
                } catch {
                    print("Error retrieving logits tensor data: \(error)")
                    continue
                }
            }
            
            beam = newBeam.sorted(by: { $0.score > $1.score }).prefix(beamWidth).map { $0 }
            print("Beam after iteration \(step): \(beam.map { $0.tokens })")
            
            if beam.allSatisfy({ $0.tokens.last == eosTokenID }) {
                print("All candidates ended with EOS. Terminating beam search early.")
                break
            }
        }
        
        if let bestCandidate = beam.sorted(by: { $0.score > $1.score }).first {
            print("Best candidate: \(bestCandidate.tokens) with score \(bestCandidate.score)")
            return bestCandidate.tokens
        } else {
            print("No valid candidate found.")
            return nil
        }
    }
    
    /// Helper to convert Data to an array of a specified type.
    private static func arrayCopiedFromData<T>(_ data: Data) -> [T]? {
        let stride = MemoryLayout<T>.stride
        print("Converting data to array of \(T.self). Data count: \(data.count) bytes, stride: \(stride) bytes.")
        guard data.count % stride == 0 else {
            print("Data size \(data.count) is not a multiple of \(stride).")
            return nil
        }
        return data.withUnsafeBytes { bytes -> [T] in
            Array(bytes.bindMemory(to: T.self))
        }
    }
}
