`ifndef N
`define N              16
`endif
`define W               8
`define lgN     ($clog2(`N))
`define dbLgN (2*$clog2(`N))

typedef struct packed { logic [`W-1:0] data; } data_t;

module add_(
    input   logic   clock,
    input   data_t  a,
    input   data_t  b,
    output  data_t  out
);
    always_ff @(posedge clock) begin
        out.data <= a.data + b.data;
    end
endmodule

module mul_(
    input   logic   clock,
    input   data_t  a,
    input   data_t  b,
    output  data_t out
);
    always_ff @(posedge clock) begin
        out.data <= a.data * b.data;
    end
endmodule

module RedUnit(
    input   logic               clock,
                                reset,
    input   data_t              data[`N-1:0],
    input   logic               split[`N-1:0],
    input   logic [`lgN-1:0]    out_idx[`N-1:0],
    output  data_t              out_data[`N-1:0],
    output  int                 delay,
    output  int                 num_el,
    output  data_t              halo_out,
    input   data_t              halo_in
);
    // num_el 总是赋值为 N
    assign num_el = `N;
    // delay 你需要自己为其赋值，表示电路的延迟
    assign delay = 1;

    data_t psum[`N-1:0];
    generate
        assign psum[0] = data[0] + halo_in;
        for(genvar i = 1; i < `N; i++) begin
            assign psum[i] = split[i-1] ? data[i] : psum[i-1] + data[i];
        end
        for (genvar i = 0; i < `N; i++) begin
            assign out_data[i] = psum[out_idx[i]];
        end
    endgenerate

    logic [`lgN-1:0] last_split;
    always_comb begin
        last_split = '{default: 0};
        halo_out = 0;
        for (int i = `N - 1; i >= 0; --i)
            if (split[i]) begin
                last_split = i;
                break;
            end
        halo_out = last_split == `N-1 ? 0 : psum[`N-1];
    end
endmodule

module PE(
    input   logic               clock,
                                reset,
    input   logic               lhs_start,
    input   logic [`dbLgN-1:0]  lhs_ptr [`N-1:0],
    input   logic [`lgN-1:0]    lhs_col [`N-1:0],
    input   data_t              lhs_data[`N-1:0],
    input   data_t              rhs[`N-1:0],
    output  data_t              out[`N-1:0],
    output  int                 delay,
    output  int                 num_el
);
    // num_el 总是赋值为 N
    assign num_el = `N;

    //State Machine
    logic valid = 0;
    assign valid = lhs_start || valid;


    // split 用于记录 lhs_ptr 的变化
    logic split[`N-1:0];
    logic [`lgN:0] current_pos_ptr;
    logic [`lgN:0] current_row_data;
    logic [`lgN:0] current_row_output;
    logic [`lgN-1:0] out_idx[`N-1:0];
    // current_row_data 记录当前data的部分和求到哪一行了
    // current_row_output 记录当前out_idx输出到哪一行了
    always_ff @(posedge clock or posedge reset) begin
        if (reset) begin
            split = '{default: 0};
            current_pos_ptr = 0;
            current_row_data = 0;
            current_row_output = 0;
            out_idx = '{default: 0};
        end else if (valid) begin
            split = '{default: 0};
            for (int i = 0; i < `N; ++i) begin
                if (lhs_ptr[current_pos_ptr] >= current_row_data * `N &&
                lhs_ptr[current_pos_ptr] < (current_row_data + 1) * `N) begin
                    split[lhs_ptr[current_pos_ptr] % `N] = 1;
                    out_idx[current_row_output++] = lhs_ptr[current_pos_ptr] % `N;
                    current_pos_ptr++;
                end
            end
            current_row_data++;
        end
    end

    // mult_result 用于记录乘法结果
    data_t mult_result[`N-1:0];
    generate
        for(genvar i = 0; i < `N; i++) begin
            mul_ multiplier(
                .clock(clock),
                .a(lhs_data[i]),
                .b(rhs[lhs_col[i]]),
                .out(mult_result[i])
            );
        end
    endgenerate

    int red_delay;
    data_t out_buffer[`N-1:0];
    data_t halo_in, halo_out;
    // Question: why negedge clock?
    always_ff @(negedge clock or posedge reset) begin
        if (reset) halo_in <= 0;
        else if (valid) halo_in <= halo_out;
    end


    RedUnit red_unit(
        .clock(clock),
        .reset(reset),
        .data(mult_result),
        .split(split),
        .out_idx(out_idx),
        .out_data(out_buffer),
        .delay(red_delay),
        .num_el(num_el),
        .halo_out(halo_out),
        .halo_in(halo_in)
    );

    logic zero[`N-1:0];
    always_comb begin
        out = '{default: 0};
        zero = '{default: 0};
        for (int i = 1; i < `N; ++i)
            if (lhs_ptr[i] == lhs_ptr[i - 1])
                zero[i] = 1;
        for (int i = 0; i < current_row_output; i++)
            if (!zero[i]) out[i] = out_buffer[i];
    end
    assign delay = red_delay;
endmodule

module SpMM(
    input   logic               clock,
                               reset,
    output  logic               lhs_ready_ns,
                               lhs_ready_ws,
                               lhs_ready_os,
                               lhs_ready_wos,
    input   logic               lhs_start,
                               lhs_ws,
                               lhs_os,
    input   logic [`dbLgN-1:0]  lhs_ptr [`N-1:0],
    input   logic [`lgN-1:0]    lhs_col [`N-1:0],
    input   data_t              lhs_data[`N-1:0],
    output  logic               rhs_ready,
    input   logic               rhs_start,
    input   data_t              rhs_data [3:0][`N-1:0],
    output  logic               out_ready,
    input   logic               out_start,
    output  data_t              out_data [3:0][`N-1:0],
    output  int                 num_el
);
    assign num_el = `N;

    //Definations

    // Input Part
    data_t rhs_buffer [`N-1:0][`N-1:0]; // RHS buffer - stores full NxN matrix
    data_t rhs_trans [`N-1:0][`N-1:0]; // RHS transposed
    logic [$clog2(`N/4)-1:0] rhs_block_counter; // Counter for blocks of 4 rows during RHS loading progress
    logic rhs_loading_done; // Signal for lhs loading
    
    // Output Part
    logic [$clog2(`N/4)-1:0] output_counter; // Counter for blocks of 4 rows during output progress
    logic output_done; // Signal for output done
    
    //Processing Part
    logic [$clog2(`N):0] processing_counter; // Counter for processing time, deciding when to output
    logic processing_done; // Signal for output ready
    int pe_delay;
    data_t pe_outputs [`N-1:0][`N-1:0]; // PE Output buffers

    


    //State Machine
    typedef enum logic [1:0] {
        IDLE,
        LOADING_RHS,
        PROCESSING,
        OUTPUT
    } state_t;
    state_t current_state, next_state;

    always_ff @(posedge clock or posedge reset) begin
        if (reset) begin
            current_state <= IDLE;
            rhs_block_counter <= 0;
            rhs_loading_done <= 0;
            processing_counter <= 0;
            processing_done <= 0;
            out_ready = 0;
            output_counter <= 0;
        end else current_state <= next_state;
    end

    // Next state logic
    always_comb begin
        next_state = current_state;
        case (current_state)
            IDLE: begin
                if (rhs_start)
                    next_state = LOADING_RHS;
            end
            LOADING_RHS: begin
                if (rhs_loading_done)
                    next_state = PROCESSING;
            end
            PROCESSING: begin
                if (processing_done)
                    next_state = OUTPUT;
            end
            OUTPUT: begin
                out_ready = 1;
                if (output_done)
                    next_state = IDLE;
            end
        endcase
    end


    // RHS Buffer Loading Logic
    always_ff @(posedge clock) begin
        if (current_state != LOADING_RHS) begin
            rhs_loading_done <= 0;
            rhs_block_counter <= 0;
        end else begin
            // Load 4 rows at a time
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < `N; j++)
                    rhs_buffer[j][rhs_block_counter * 4 + i] <= rhs_data[i][j];
            
            // Update counter
            if (rhs_block_counter < (`N/4) - 1) rhs_block_counter <= rhs_block_counter + 1;
            else rhs_loading_done <= 1;
        end 
    end

    // PE array instantiation - N PEs, each processing one column
    genvar i;
    generate
        for (i = 0; i < `N; i++) begin : pe_array
            PE pe_inst (
                .clock(clock),
                .reset(reset),
                .lhs_start(lhs_start && current_state == PROCESSING),
                .lhs_ptr(lhs_ptr),
                .lhs_col(lhs_col),
                .lhs_data(lhs_data),
                .rhs(rhs_buffer[i]),
                .out(pe_outputs[i]),
                .delay(pe_delay),
                .num_el()
            );
        end
    endgenerate

    always_ff @(posedge clock) begin
        if (current_state != PROCESSING) begin  
            processing_counter <= 0;
            processing_done <= 0;
        end else begin
            if (processing_counter < (`N + pe_delay)) 
                processing_counter <= processing_counter + 1;
            else processing_done <= 1;
        end
    end


    //Output Buffer Logic
    always_ff @(posedge clock or posedge reset or posedge out_ready) begin
        if (current_state != OUTPUT) begin
            out_ready = 0;
            output_done = 0;
            output_counter <= 0;
        end else begin
            for (int block = 0; block < 4; block++)
                for (int col = 0; col < `N; col++)
                    assign out_data[block][col] = pe_outputs[col][4 * output_counter + block];

            if (output_counter < (`N/4) - 1) output_counter <= output_counter + 1;
            else output_done = 1;
        end
    end

    // Control signals
    assign rhs_ready = (current_state == IDLE);
    assign lhs_ready_ns = (current_state == PROCESSING);

    // Not implemented yet
    assign lhs_ready_ws = 0;
    assign lhs_ready_os = 0;
    assign lhs_ready_wos = 0;

endmodule