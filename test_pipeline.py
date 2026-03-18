import argparse
import sys
import time

try:
    import httpx
except ImportError:
    print("Error: 'httpx' is required. Install it using 'pip install httpx'")
    sys.exit(1)

def test_pipeline(audio_file_path: str, output_file_path: str, session_id: str = "test-session"):
    print(f"🎙️ Sending '{audio_file_path}' to the API for processing...")
    
    url = "http://localhost:8000/voice/process"
    
    try:
        with open(audio_file_path, "rb") as f:
            files = {"audio": (audio_file_path, f, "audio/wav")}
            data = {"session_id": session_id}
            
            start_time = time.time()
            # The timeout is high because CPU inference for LLM and TTS can take some time
            response = httpx.post(url, files=files, data=data, timeout=300.0)
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                print(f"\n✅ Success! Processing took {elapsed:.1f} seconds.")
                
                # The headers contain the STT and LLM results
                transcript = response.headers.get("X-Transcript", "N/A")
                reply_text = response.headers.get("X-Reply-Text", "N/A")
                
                print(f"\n📝 STT Transcript: {transcript}")
                print(f"🤖 LLM Reply: {reply_text}")
                
                # The body contains the TTS audio bytes
                with open(output_file_path, "wb") as out_f:
                    out_f.write(response.content)
                print(f"\n🔊 Output audio saved to: {output_file_path}")
            else:
                print(f"❌ Error: {response.status_code}")
                print(response.text)
                
    except FileNotFoundError:
        print(f"❌ Could not find audio file at '{audio_file_path}'")
    except httpx.RequestError as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Voice Bot Pipeline")
    parser.add_argument("--input", default="ljspeech_model/reference.wav", help="Input WAV file")
    parser.add_argument("--output", default="pipeline_test_response.wav", help="Where to save the output WAV")
    parser.add_argument("--session", default="demo", help="Session ID for LLM history")
    
    args = parser.parse_args()
    test_pipeline(args.input, args.output, args.session)
