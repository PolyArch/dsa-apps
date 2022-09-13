#include "../Common/spatial_inrin.h"
#include "../Common/test.h"
#include "../Common/interface.h"
#include "../Specs/backprop.h"

// void soft_max(TYPE net_outputs[possible_outputs], TYPE activations[possible_outputs]) {
//   int i;
//   TYPE sum;
//   sum = (TYPE) 0.0;
// 
//   #pragma ss config
//   {
//     #pragma ss stream
//     #pragma ss dfg dedicated unroll(2)
//     for(i=0; i < possible_outputs; i++) {
//         // sum += exp64(-activations[i]);
//         sum += activations[i];
//     }
//     #pragma ss stream
//     #pragma ss dfg dedicated unroll(2)
//     for(i=0; i < possible_outputs; i++) {
//         net_outputs[i] = activations[i] / sum;
//     }
//   }
// }

// void RELU(TYPE *__restrict activations, TYPE *__restrict dactivations, int64_t size) {
//   #pragma ss config
//   {
//     int64_t i;
//     #pragma ss stream
//     #pragma ss dfg dedicated unroll(2)
//     for( i = 0; i < size; i++) {
//         dactivations[i] = activations[i] * (1.0 - activations[i]);
//         // activations[i] = 1.0/(1.0+exp(-activations[i]));
//         activations[i] = 1.0 / (1.0 + activations[i]);
//     }
//   }
// }

void matrix_vector_product_with_bias_input_layer(TYPE biases[nodes_per_layer],
                                                 TYPE weights[input_dimension*nodes_per_layer],
                                                 TYPE activations[nodes_per_layer],
                                                 TYPE input_sample[input_dimension],
                                                 TYPE dactivations[nodes_per_layer]) {
  #pragma ss config
  {
    arrayhint(biases, nodes_per_layer * sizeof(TYPE), 1 - 1.0 / training_sets);
    arrayhint(weights, input_dimension * nodes_per_layer * sizeof(TYPE), 1 - 1.0 / training_sets);
    arrayhint(activations, nodes_per_layer * sizeof(TYPE), 1.0 / training_sets / nodes_per_layer);
    arrayhint(input_sample, input_dimension * sizeof(TYPE), 1 - 1.0 / nodes_per_layer);
    arrayhint(dactivations, nodes_per_layer * sizeof(TYPE), 1.0 / training_sets);
    TYPE one = 1.0;
    TYPE spad[input_dimension];
    #pragma ss stream
    #pragma ss dfg dedicated unroll(2)
    for (int64_t i = 0; i < input_dimension; ++i) {
      spad[i] = input_sample[i];
    }
    #pragma ss stream
    for (int64_t j = 0; j < nodes_per_layer; j++) {
        TYPE acc = (TYPE)0.0;
        #pragma ss dfg dedicated unroll(4)
        for (int64_t i = 0; i < input_dimension; i++){
            acc += weights[j*input_dimension + i] * spad[i];
        }
        TYPE act = acc + biases[j];
        activations[j] = one / (one + act);
        dactivations[j] = act * (one - act);
    }
  }
}

void matrix_vector_product_with_bias_second_layer(TYPE biases[nodes_per_layer],
                                                  TYPE weights[nodes_per_layer*nodes_per_layer],
                                                  TYPE activations[nodes_per_layer],
                                                  TYPE input_activations[nodes_per_layer],
                                                  TYPE dactivations[nodes_per_layer]){
  #pragma ss config
  {
    arrayhint(biases, nodes_per_layer * sizeof(TYPE), 1 - 1.0 / training_sets);
    arrayhint(weights, nodes_per_layer * nodes_per_layer * sizeof(TYPE), 1 - 1.0 / training_sets);
    arrayhint(activations, nodes_per_layer * sizeof(TYPE), 1 - 1.0 / training_sets / nodes_per_layer);
    arrayhint(input_activations, nodes_per_layer * sizeof(TYPE), 1 - 1.0 / training_sets);
    arrayhint(dactivations, nodes_per_layer * sizeof(TYPE), 1 - 1.0 / training_sets);
    TYPE one = 1.0;
    TYPE spad[input_dimension];
    #pragma ss stream
    #pragma ss dfg dedicated unroll(2)
    for (int64_t i = 0; i < input_dimension; ++i) {
      spad[i] = input_activations[i];
    }
    #pragma ss stream
    for (int64_t i = 0; i < nodes_per_layer; i++){
        TYPE acc = (TYPE)0.0;
        #pragma ss dfg dedicated unroll(4)
        for(int64_t j = 0; j < nodes_per_layer; j++){
            acc += weights[i*nodes_per_layer + j] * spad[j];
        }
        TYPE act = acc + biases[i];
        activations[i] = one / (one + act);
        dactivations[i] = act * (one - act);
    }
  }
}
 
void matrix_vector_product_with_bias_output_layer(TYPE biases[possible_outputs],
                                                  TYPE weights[nodes_per_layer*possible_outputs],
                                                  TYPE activations[possible_outputs],
                                                  TYPE input_activations[nodes_per_layer],
                                                  TYPE dactivations[possible_outputs],
                                                  TYPE net_outputs[possible_outputs],
                                                  TYPE solutions[possible_outputs],
                                                  TYPE output_difference[possible_outputs]){
  #pragma ss config
  {
    // arrayhint(biases, possible_outputs * sizeof(TYPE), 1 - 1.0 / training_sets);
    arrayhint(weights, possible_outputs * sizeof(TYPE), 1 - 1.0 / training_sets);
    // arrayhint(activations, possible_outputs * sizeof(TYPE), 1 - 1.0 / training_sets / nodes_per_layer);
    arrayhint(input_activations, nodes_per_layer * sizeof(TYPE), 1 - 1.0 / training_sets);
    // arrayhint(dactivations, possible_outputs * sizeof(TYPE), 1 - 1.0 / training_sets);
    // arrayhint(net_outputs, possible_outputs * sizeof(TYPE), 1 - 1.0 / training_sets);
    // arrayhint(output_difference, possible_outputs * sizeof(TYPE), 1 - 1.0 / training_sets);
    TYPE one = 1.0;
    TYPE acc0 = (TYPE)0.0;
    TYPE acc1 = (TYPE)0.0;
    TYPE acc2 = (TYPE)0.0;
    #pragma ss stream
    #pragma ss dfg dedicated unroll(2)
    for (int64_t i = 0; i < nodes_per_layer; i++){
      acc0 += weights[0*nodes_per_layer + i] * input_activations[i];
      acc1 += weights[1*nodes_per_layer + i] * input_activations[i];
      acc2 += weights[2*nodes_per_layer + i] * input_activations[i];
    }
    //
    TYPE act0 = acc0 + biases[0];
    TYPE act1 = acc1 + biases[1];
    TYPE act2 = acc2 + biases[2];
    activations[0] = act0 / (one + act0);
    activations[1] = act1 / (one + act1);
    activations[2] = act2 / (one + act2);
    dactivations[0] = act0 * (one - act0);
    dactivations[1] = act0 * (one - act1);
    dactivations[2] = act0 * (one - act2);
    TYPE sum = act0 + act1 + act2;
    net_outputs[0] = act0 / sum;
    net_outputs[1] = act1 / sum;
    net_outputs[2] = act2 / sum;
    output_difference[0] = ((solutions[0] - (net_outputs[0]))) * dactivations[0];
    output_difference[1] = ((solutions[1] - (net_outputs[1]))) * dactivations[1];
    output_difference[2] = ((solutions[2] - (net_outputs[2]))) * dactivations[2];
  }
}

// void take_difference(TYPE net_outputs[possible_outputs], 
//                      TYPE solutions[possible_outputs], 
//                      TYPE output_difference[possible_outputs],
//                      TYPE dactivations[possible_outputs]) {
//   #pragma ss config
//   {
//     int i;
//     #pragma ss stream
//     #pragma ss dfg dedicated unroll(4)
//     for( i = 0; i < possible_outputs; i++){
//         output_difference[i] = ((solutions[i] - (net_outputs[i]))) * dactivations[i];
//     }
//   }
// }

void get_delta_matrix_weights3(TYPE delta_weights3[nodes_per_layer*possible_outputs],
                               TYPE output_difference[possible_outputs],
                               TYPE last_activations[nodes_per_layer]) {
  #pragma ss config
  {
    arrayhint(delta_weights3, nodes_per_layer * possible_outputs * sizeof(TYPE), 1 - 1.0 / training_sets);
    // arrayhint(output_difference, possible_outputs * sizeof(TYPE), 1 - 1.0 / training_sets / 2.0);
    arrayhint(last_activations, nodes_per_layer * sizeof(TYPE), 1 - 1.0 / training_sets);
    #pragma ss stream
    #pragma ss dfg dedicated unroll(1)
    for(int64_t i = 0; i < nodes_per_layer; i++) {
      delta_weights3[i * possible_outputs + 0] = last_activations[i] * output_difference[0];
      delta_weights3[i * possible_outputs + 1] = last_activations[i] * output_difference[1];
      delta_weights3[i * possible_outputs + 2] = last_activations[i] * output_difference[2];
    }
  }
}

void get_oracle_activations2(TYPE *__restrict weights3, 
                             TYPE *__restrict output_differences, 
                             TYPE *__restrict oracle_activations,
                             TYPE *__restrict dactivations) {
  #pragma ss config
  {
    arrayhint(weights3, nodes_per_layer * possible_outputs * sizeof(TYPE), 1.0 - 1.0 / training_sets);
    // arrayhint(output_differences, possible_outputs * sizeof(TYPE), 1 - 1.0 / training_sets / 2.0);
    arrayhint(oracle_activations, nodes_per_layer * sizeof(TYPE), 1.0 - 1.0 / training_sets);
    arrayhint(dactivations, nodes_per_layer * sizeof(TYPE), 1.0 - 1.0 / training_sets);
    #pragma ss stream
    #pragma ss dfg dedicated
    for(int64_t i = 0; i < nodes_per_layer; i++) {
      TYPE acc = 0.0;
      acc += output_differences[0] * weights3[i*possible_outputs + 0];
      acc += output_differences[1] * weights3[i*possible_outputs + 1];
      acc += output_differences[2] * weights3[i*possible_outputs + 2];
      oracle_activations[i] = acc * dactivations[i];
    }
  }
}

void get_delta_matrix_weights2(TYPE delta_weights2[nodes_per_layer*nodes_per_layer],
                               TYPE output_difference[nodes_per_layer],
                               TYPE last_activations[nodes_per_layer]) {
  #pragma ss config
  {
    arrayhint(delta_weights2, nodes_per_layer * nodes_per_layer * sizeof(TYPE), 1 - 1.0 / training_sets / nodes_per_layer);
    arrayhint(output_difference, nodes_per_layer * sizeof(TYPE), 1 - 1.0 / training_sets / nodes_per_layer);
    arrayhint(last_activations, nodes_per_layer * sizeof(TYPE), 1 - 1.0 / training_sets);
    TYPE spad[nodes_per_layer];
    #pragma ss stream
    #pragma ss dfg dedicated unroll(2)
    for(int64_t j = 0; j < nodes_per_layer; j++) {
        spad[j] = output_difference[j];
    }
    #pragma ss stream
    for(int64_t i = 0; i < nodes_per_layer; i++) {
      #pragma ss dfg dedicated unroll(4)
      for(int64_t j = 0; j < nodes_per_layer; j++) {
          delta_weights2[i*nodes_per_layer + j] = last_activations[i] * output_difference[j];
      }
    }
  }
}

void get_oracle_activations1(TYPE weights2[nodes_per_layer*nodes_per_layer], 
                             TYPE output_differences[nodes_per_layer], 
                             TYPE oracle_activations[nodes_per_layer],
                             TYPE dactivations[nodes_per_layer]) {
  #pragma ss config
  {
    arrayhint(weights2, nodes_per_layer * nodes_per_layer * sizeof(TYPE), 1.0 - 1.0 / training_sets);
    arrayhint(output_differences, nodes_per_layer * sizeof(TYPE), 1.0 - 1.0 / training_sets / nodes_per_layer);
    arrayhint(oracle_activations, nodes_per_layer * sizeof(TYPE), 1.0 - 1.0 / training_sets);
    arrayhint(dactivations, nodes_per_layer * sizeof(TYPE), 1.0 - 1.0 / training_sets);
    TYPE spad[nodes_per_layer];
    #pragma ss stream
    #pragma ss dfg dedicated unroll(2)
    for(int64_t j = 0; j < nodes_per_layer; j++) {
        spad[j] = output_differences[j];
    }
    #pragma ss stream
    for(int64_t i = 0; i < nodes_per_layer; i++) {
      double acc = 0.0;
      #pragma ss dfg dedicated unroll(4)
      for(int64_t j = 0; j < nodes_per_layer; j++) {
        acc += output_differences[j] * weights2[i*nodes_per_layer + j];
      }
      oracle_activations[i] = acc * dactivations[i];
    }
  }
}

void get_delta_matrix_weights1(TYPE delta_weights1[input_dimension*nodes_per_layer],
                               TYPE output_difference[nodes_per_layer],
                               TYPE last_activations[input_dimension]) {
  #pragma ss config
  {
    arrayhint(delta_weights1, input_dimension * nodes_per_layer * sizeof(TYPE), 1.0 - 1.0 / training_sets);
    arrayhint(output_difference, nodes_per_layer * sizeof(TYPE), 1.0 - 1.0 / training_sets / input_dimension);
    arrayhint(last_activations, input_dimension * sizeof(TYPE), 1.0 - 1.0 / training_sets);
    #pragma ss stream
    for(int64_t i = 0; i < input_dimension; i++) {
      #pragma ss dfg dedicated unroll(4)
      for(int64_t j = 0; j < nodes_per_layer; j++) {
          delta_weights1[i*nodes_per_layer + j] = last_activations[i] * output_difference[j];
      }
    }
  }
}

void update_weights(TYPE *__restrict weights1,
                    TYPE *__restrict weights2,
                    TYPE *__restrict weights3,
                    TYPE *__restrict d_weights1,
                    TYPE *__restrict d_weights2,
                    TYPE *__restrict d_weights3,
                    TYPE *__restrict biases1,
                    TYPE *__restrict biases2,
                    TYPE *__restrict biases3,
                    TYPE *__restrict d_biases1,
                    TYPE *__restrict d_biases2,
                    TYPE *__restrict d_biases3) {
  double reuse = 1.0 - 1.0 / training_sets / 2.0;
  #pragma ss config
  {
    arrayhint(weights1, input_dimension*nodes_per_layer * sizeof(TYPE), reuse);
    arrayhint(d_weights1, input_dimension*nodes_per_layer * sizeof(TYPE), reuse);
    TYPE norm;
    norm = 0.0;
    #pragma ss stream nonblock
    for (int64_t i=0; i < input_dimension; i++){
      #pragma ss dfg dedicated unroll(4)
      for (int64_t j = 0; j < nodes_per_layer; j++){
        double x = (weights1[i*nodes_per_layer + j] - d_weights1[i*nodes_per_layer + j] * 1e-2);
        norm += x * x;
        weights1[i*nodes_per_layer + j] = x;
      }
    }
    #pragma ss dfg temporal
    {
      norm = 1.0 / fsqrt(norm);
    }
    #pragma ss stream
    for (int64_t i=0; i < input_dimension; i++){
      #pragma ss dfg dedicated unroll(4)
      for (int64_t j = 0; j < nodes_per_layer; j++){
        weights1[i*nodes_per_layer + j] = (weights1[i*nodes_per_layer + j] * norm);
      }
    }
  }

  #pragma ss config
  {
    arrayhint(biases1, nodes_per_layer * sizeof(TYPE), reuse);
    arrayhint(d_biases1, nodes_per_layer * sizeof(TYPE), reuse);
    TYPE bias_norm = 0;
    #pragma ss stream nonblock
    #pragma ss dfg dedicated unroll(4)
    for(int64_t i=0; i < nodes_per_layer; i++){
      double x = biases1[i] - d_biases1[i] * 1e-2;
      bias_norm += x * x;
      biases1[i] = x;
    }
    #pragma ss dfg temporal
    {
      bias_norm = 1.0 / fsqrt(bias_norm);
    }
    #pragma ss stream
    #pragma ss dfg dedicated unroll(4)
    for(int64_t i=0; i < nodes_per_layer; i++){
        biases1[i] = (biases1[i] * bias_norm);
    }
  }

  #pragma ss config
  {
    arrayhint(weights2, nodes_per_layer * nodes_per_layer * sizeof(TYPE), reuse);
    arrayhint(d_weights2, nodes_per_layer * nodes_per_layer * sizeof(TYPE), reuse);
    double norm = (double)0.0;
    #pragma ss stream nonblock
    for(int64_t i=0; i < nodes_per_layer; i++){
      #pragma ss dfg dedicated unroll(4)
      for(int64_t j = 0; j < nodes_per_layer; j++){
          double x = weights2[i*nodes_per_layer + j] - (d_weights2[i*nodes_per_layer + j] * 1e-2);
          norm += x * x;
          weights2[i*nodes_per_layer + j] = x;
      }
    }
    #pragma ss dfg temporal
    {
      norm = fsqrt(1.0 / norm);
    }
    #pragma ss stream
    for(int64_t i=0; i < nodes_per_layer; i++){
      #pragma ss dfg dedicated unroll(4)
      for(int64_t j = 0; j < nodes_per_layer; j++){
          weights2[i*nodes_per_layer + j] = (weights2[i*nodes_per_layer + j] * norm);
      }
    }
  }

  #pragma ss config
  {
    arrayhint(biases2, nodes_per_layer * sizeof(TYPE), reuse);
    arrayhint(d_biases2, nodes_per_layer * sizeof(TYPE), reuse);
    TYPE bias_norm = 0.0;
    #pragma ss stream nonblock
    #pragma ss dfg dedicated unroll(4)
    for(int64_t i = 0; i < nodes_per_layer; i++){
        double x = biases2[i] - (d_biases2[i] * 1e-2);
        bias_norm += x * x;
        biases2[i] = x;
    }
    #pragma ss dfg temporal
    {
      bias_norm = fsqrt(1.0 / bias_norm);
    }
    #pragma ss stream
    #pragma ss dfg dedicated unroll(4)
    for(int64_t i = 0; i < nodes_per_layer; i++){
        biases2[i] = (biases2[i] * bias_norm);
    }
  }

  #pragma ss config
  {
    arrayhint(weights3, nodes_per_layer * possible_outputs * sizeof(TYPE), reuse);
    arrayhint(d_weights3, nodes_per_layer * possible_outputs * sizeof(TYPE), reuse);
    TYPE norm = 0.0;
    #pragma ss stream nonblock
    #pragma ss dfg dedicated
    for(int64_t i = 0; i < nodes_per_layer; i++){
      TYPE v0 = weights3[i * possible_outputs + 0];
      TYPE v1 = weights3[i * possible_outputs + 1];
      TYPE v2 = weights3[i * possible_outputs + 2];
      v0 -= (d_weights3[i * possible_outputs + 0] * 1e-2);
      v1 -= (d_weights3[i * possible_outputs + 1] * 1e-2);
      v2 -= (d_weights3[i * possible_outputs + 2] * 1e-2);
      TYPE a = v0 * weights3[i*possible_outputs + 0];
      TYPE b = v1 * weights3[i*possible_outputs + 1];
      TYPE c = v2 * weights3[i*possible_outputs + 2];
      TYPE d = a + b;
      TYPE e = c + d;
      norm += e;
      weights3[i*possible_outputs + 0] = v0;
      weights3[i*possible_outputs + 1] = v1;
      weights3[i*possible_outputs + 2] = v2;
    }
    #pragma ss dfg temporal
    {
      norm = 1.0 / fsqrt(norm);
    }
    #pragma ss stream
    for(int64_t i=0; i < nodes_per_layer; i++){
      weights3[i*possible_outputs + 0] = (weights3[i*possible_outputs + 0] * norm);
      weights3[i*possible_outputs + 1] = (weights3[i*possible_outputs + 1] * norm);
      weights3[i*possible_outputs + 2] = (weights3[i*possible_outputs + 2] * norm);
    }
  }

  #pragma ss config
  {
    arrayhint(biases3, possible_outputs * sizeof(TYPE), reuse);
    arrayhint(d_biases3, possible_outputs * sizeof(TYPE), reuse);
    int64_t i;
    double bias_norm = 0;
    #pragma ss stream nonblock
    #pragma ss dfg dedicated unroll(4)
    for(i=0; i < possible_outputs;i++){
        TYPE v0 = biases3[i];
        v0 -= d_biases3[i] * 1e-2;
        bias_norm += v0 * v0;
        biases3[i] = v0;
    }
    #pragma ss dfg temporal
    {
      bias_norm = 1.0 / fsqrt(bias_norm);
    }
    #pragma ss stream
    #pragma ss dfg dedicated unroll(4)
    for(i=0; i < possible_outputs; i++){
        biases3[i] = (biases3[i] * bias_norm);
    }
  }
}

//Forward and training structures
TYPE activations1[nodes_per_layer];
TYPE activations2[nodes_per_layer];
TYPE activations3[possible_outputs];
TYPE dactivations1[nodes_per_layer];
TYPE dactivations2[nodes_per_layer];
TYPE dactivations3[possible_outputs];
TYPE net_outputs[possible_outputs];
//Training structure
TYPE output_difference[possible_outputs];
TYPE delta_weights1[input_dimension*nodes_per_layer]; 
TYPE delta_weights2[nodes_per_layer*nodes_per_layer];
TYPE delta_weights3[nodes_per_layer*possible_outputs];
TYPE oracle_activations1[nodes_per_layer];
TYPE oracle_activations2[nodes_per_layer];

void backprop(TYPE weights1[input_dimension*nodes_per_layer], 
              TYPE weights2[nodes_per_layer*nodes_per_layer],
              TYPE weights3[nodes_per_layer*possible_outputs],
              TYPE biases1[nodes_per_layer], 
              TYPE biases2[nodes_per_layer],
              TYPE biases3[possible_outputs],
              TYPE training_data[training_sets*input_dimension],
              TYPE training_targets[training_sets*possible_outputs]) {

    for(int64_t i = 0; i < training_sets; i++) {
        // for(j=0;j<nodes_per_layer;j++){
        //     activations1[j] = (TYPE)0.0;
        //     activations2[j] = (TYPE)0.0;
        //     if(j<possible_outputs){
        //         activations3[j] = (TYPE)0.0;
        //     }
        // }
        matrix_vector_product_with_bias_input_layer(
          biases1, weights1, activations1, &training_data[i*input_dimension], dactivations1);
        matrix_vector_product_with_bias_second_layer(
          biases2, weights2, activations2, activations1, dactivations2);
        matrix_vector_product_with_bias_output_layer(
          biases3, weights3, activations3, activations2, dactivations3, net_outputs,
          &training_targets[i*possible_outputs], output_difference);
        get_delta_matrix_weights3(delta_weights3, output_difference, activations2);
        get_oracle_activations2(weights3, output_difference, oracle_activations2, dactivations2);
        get_delta_matrix_weights2(delta_weights2, oracle_activations2, activations1);
        get_oracle_activations1(weights2, oracle_activations2, oracle_activations1, dactivations1);
        get_delta_matrix_weights1(delta_weights1, oracle_activations1, &training_data[i*input_dimension]);
        update_weights(weights1, weights2, weights3, delta_weights1, delta_weights2, delta_weights3, 
                      biases1, biases2, biases3, oracle_activations1, oracle_activations2, output_difference);
    }
}

struct Arguments {
  TYPE weights1[input_dimension*nodes_per_layer]; 
  TYPE weights2[nodes_per_layer*nodes_per_layer];
  TYPE weights3[nodes_per_layer*possible_outputs];
  TYPE biases1[nodes_per_layer]; 
  TYPE biases2[nodes_per_layer];
  TYPE biases3[possible_outputs];
  TYPE training_data[training_sets*input_dimension];
  TYPE training_targets[training_sets*possible_outputs];
} args_;

NO_INIT_DATA
NO_SANITY_CHECK

void run_accelerator(struct Arguments *args, int _) {
  backprop(args->weights1, args->weights2, args->weights3,
           args->biases1, args->biases2, args->biases3,
           args->training_data,  args->training_targets);
}
