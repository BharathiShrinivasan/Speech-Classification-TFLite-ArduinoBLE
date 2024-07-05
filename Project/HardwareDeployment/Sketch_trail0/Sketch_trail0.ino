#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "AudioInputTensor.h"
#include "SC_LiteModel_toDeploy.h"

// Declare tflite Interpreter, input output tensor areana
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
tflite::ErrorReporter* reporter = nullptr;
TfLiteTensor* input_ptr = nullptr;
TfLiteTensor* output_ptr = nullptr;
constexpr int kTensorArenaSize = 200000; //  pick a big enough number
uint8_t tensor_arena[ kTensorArenaSize ] = { 0 };
float* input_buffer_ptr = nullptr;

/*******************************************************************************************************************************************************************************************/

void PlaceInputTensor(float* dest_ptr, const float* audio_ptr){
  for(int i=0;i<64*64;i++){ // the tensor shape is (1,64,64,1) or 64*64 elements
    dest_ptr[i] = audio_ptr[i];
  }
}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  while (!Serial);
  Serial.println("Started Serial Monitor");

  // Load Model
  static tflite::MicroErrorReporter error_reporter;
  reporter = &error_reporter;
  reporter->Report( "Speech Classification | Class=(Forward, Reverse, Unknown) " );

  model = tflite::GetModel( tf_model ); // reads the SC_LiteModel
  if( model->version() != TFLITE_SCHEMA_VERSION ) {
    reporter->Report( "Model is schema version: %d\nSupported schema version is: %d", model->version(), TFLITE_SCHEMA_VERSION );
    return;
  }
  
  // Setup our TF runner / interpreter
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, reporter );
  interpreter = &static_interpreter;
  
  
  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if( allocate_status != kTfLiteOk ) {
    reporter->Report( "AllocateTensors() failed" );
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input_ptr = interpreter->input(0);
  output_ptr = interpreter->output(0);

  // Save the input buffer to put our Audio Tensor
  input_buffer_ptr = input_ptr->data.f;  

}

void loop() {
  // put your main code here, to run repeatedly:
  Serial.println("Start Inference :");
  
  Serial.println("1. Placing Audio Tensor -> Interpreter's input");
  PlaceInputTensor(input_buffer_ptr,Audio_0);
  
  // Run our model
  Serial.print("2. Invoke Interpreter");
  TfLiteStatus invoke_status = interpreter->Invoke();
  if( invoke_status != kTfLiteOk ) {
    reporter->Report( "Invoke failed" );
    return;
  }
  else Serial.println(" - ok");
  
  // Interpret output
  Serial.println("3. Read the Label tensor -> Interpreter's output");
  float* result = output_ptr->data.f;
  Serial.print("Prediction - Output tensor :(");Serial.print(result[0]);Serial.print(", ");Serial.print(result[1]);Serial.print(", ");Serial.print(result[2]);Serial.println(")");
  switch(std::distance( result, std::max_element( result, result + 3 ) )){
    case 0: Serial.println("Prediction - Label : Forward command"); break;
    case 1: Serial.println("Prediction - Label : Reverse command"); break;
    case 2: Serial.println("Prediction - Label : Unknown command"); break;
    default: Serial.println("Prediction - Label : xx"); break;
  }
  
  Serial.println("4. Done");
  Serial.println();
  delay( 10000 );
}
