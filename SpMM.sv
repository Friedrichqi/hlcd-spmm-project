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

module RedUnit1(
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
    input   data_t              halo_in,
    input   logic               zero[`N-1:0],
    input   logic [`lgN-1:0]    out_scale[2]
);
    data_t              out_data_dummy[`N-1:0];
    localparam NUM_LEVELS = $clog2(`N);
    //vecID 表示这一组待累加的数据属于第几行
    logic [NUM_LEVELS:0] vecID [NUM_LEVELS:0][`N-1:0];
    int adderLvl [NUM_LEVELS:0][`N-1:0];
    logic add_En[NUM_LEVELS:0][`N-1:0];
    logic bypass_En[NUM_LEVELS:0][`N-1:0];
    logic [NUM_LEVELS:0] left_sel[NUM_LEVELS:0][`N-1:0];
    logic [NUM_LEVELS:0] vecid[`N-1:0];
    generate
        for(genvar i = 0; i < `N-1; i ++)
            assign vecid[i] = vecID[0][i];
    endgenerate
    generate
        for(genvar i = 0; i < `N-1; i ++)begin
            wire addEn_probe;
            assign addEn_probe = add_En[0][i];
            wire bypassEn_probe;
            assign bypassEn_probe = bypass_En[0][i];
            wire [NUM_LEVELS:0] lsel;
            assign lsel = left_sel[0][i];
            wire [NUM_LEVELS:0] rsel;
            assign rsel = right_sel[0][i];

        end
    endgenerate
    logic [NUM_LEVELS:0] right_sel[NUM_LEVELS:0][`N-1:0];
    //split_register 表示FAN network最后一行数据是否需要分裂
    logic split_register [NUM_LEVELS:0][`N-1:0];
    //out_idx_register 表示FAN network最后一行数据的输出位置
    logic [`lgN-1:0] out_idx_register[NUM_LEVELS:0][`N-1:0];
    logic zero_register[NUM_LEVELS:0][`N-1:0];
    logic [`lgN-1:0] out_scale_register[NUM_LEVELS:0][2];
    always_ff @(posedge clock or posedge reset) begin
        if (reset) begin
            for (int i = 0; i < `N; i++) begin
                vecID[0][i] = 0;
                add_En[0][i] = 0;
                bypass_En[0][i] = 0;
                adderLvl[0][i] = 0;
                left_sel[0][i] = 0;
                right_sel[0][i] = 0;
                split_register[0][i] = 0;
                out_idx_register[0][i] = 0;
                zero_register[0][i] = 0;
            end
            out_scale_register[0][0] = 0;
            out_scale_register[0][1] = `N-1;
        end else begin
            split_register[0] = split;
            out_idx_register[0] = out_idx;
            out_scale_register[0] = out_scale;
            zero_register[0] = zero;
            for (int i = 1; i < `N; i++) vecID[0][i] = split[i - 1] ? vecID[0][i - 1] + 1 : vecID[0][i - 1];
            
            for (int i = 0; i < `N; i++) adderLvl[0][i] = 0;
            for (int level = 0; level < NUM_LEVELS; level++) begin
                int step = 2 << level;
                for (int j = (1 << level) - 1; j < `N; j += step) adderLvl[0][j] = level;
            end

            for (int i = 0; i < `N; i++) begin
                add_En[0][i] = 0;
                bypass_En[0][i] = 0;
            end
            for (int i = 0; i < `N; i++)
                if (i + 1 < `N && vecID[0][i] == vecID[0][i + 1]) add_En[0][i] = 1;
                else if (adderLvl[0][i] == 0) bypass_En[0][i] = 1;

            for (int i = 0; i < `N; i++) begin
                if (adderLvl[0][i] > 0) begin
                    left_sel[0][i] = i - (1 << (adderLvl[0][i] - 1));
                    right_sel[0][i] = i + (1 << (adderLvl[0][i] - 1));

                    for (int j = 1; j < adderLvl[0][i]; j++) begin
                        if (i - (1 << j) >= 0 && vecID[0][i - (1 << j)] != vecID[0][i]) begin
                            left_sel[0][i] = i - j;
                            break;
                        end
                    end
                    for (int j = 1; j < adderLvl[0][i]; j++) begin
                        if (i + (1 << j) + 1 < `N && vecID[0][i + (1 << j) + 1] != vecID[0][i]) begin
                            right_sel[0][i] = i + j;
                            break;
                        end
                    end
                end
            end
        end
    end

    //fan_data 表示FAN network中每一行加法器中的数据
    data_t fan_data[NUM_LEVELS:0][`N-1:0];
    //fan_register 表示FAN network中上一行处理完的数据（输出），作为下一行的输入
    data_t fan_register[NUM_LEVELS:0][`N-1:0];
    data_t final_fan_register [`N-1:0];
    generate
        for (genvar i = 0  ; i < `N; i++ ) begin
            assign final_fan_register[i] = fan_register[NUM_LEVELS-1][i];
        end
    endgenerate

    //fan_out_idx 表示这一行数据在FAN network结构中的的输出位置
    logic [NUM_LEVELS:0] fan_out_idx[NUM_LEVELS:0][`N-1:0];
    //fan_out_idx_register 表示FAN network中上一行处理完的数据的输出位置，在下一行处理完后，可能会更新到这一组数据中最深的累加器/register编号
    logic [NUM_LEVELS:0] fan_out_idx_register[NUM_LEVELS:0][`N-1:0];
    //out_data_register 表示FAN network中最后一行的输出数据转换成prefix结构的输出数据
    data_t out_data_register[`N-1:0];
    int flag = 0;
    generate
        for (genvar level = 0; level < NUM_LEVELS; level++) begin : gen_level
            localparam step = 2 << level;
            always_ff @(posedge clock or posedge reset) begin
                if (reset) begin
                    for (int i = 0; i < `N; i++) begin
                        fan_data[level+1][i] <= 0;
                        vecID[level+1][i] <= 0;
                        add_En[level+1][i] <= 0;
                        bypass_En[level+1][i] <= 0;
                        left_sel[level+1][i] <= 0;
                        right_sel[level+1][i] <= 0;
                        split_register[level+1][i] <= 0;
                        out_idx_register[level+1][i] <= 0;
                        zero_register[level+1][i] <= 0;
                    end
                    out_scale_register[level+1][0] <= 0;
                    out_scale_register[level+1][1] <= `N-1;
                end else begin
                    if (level == 0) begin
                        for (int i = 0; i < `N; i += step) begin
                            if (add_En[level][i]) begin
                                fan_data[level+1][i] <= data[i] + data[i + 1];
                                fan_out_idx[level+1][vecID[level][i]] <= i;
                            end else if (bypass_En[level][i]) begin
                                fan_data[level+1][i] <= data[i];
                                fan_data[level+1][i + 1] <= data[i + 1];
                                fan_out_idx[level+1][vecID[level][i]] <= i;
                                fan_out_idx[level+1][vecID[level][i + 1]] <= i + 1;
                            end
                        end
                        for (int i = 0; i < `N; i++) begin
                            zero_register[level+1][i] <= zero_register[level][i];
                            out_idx_register[level+1][i] <= out_idx_register[level][i];
                            split_register[level+1][i] <= split_register[level][i];
                            vecID[level+1][i] <= vecID[level][i];
                            add_En[level+1][i] <= add_En[level][i];
                            bypass_En[level+1][i] <= bypass_En[level][i];
                            adderLvl[level+1][i] <= adderLvl[level][i];
                            left_sel[level+1][i] <= left_sel[level][i];
                            right_sel[level+1][i] <= right_sel[level][i];
                        end
                        out_scale_register[level+1] <= out_scale_register[level];
                    end
                    else begin
                        for (int i = 0; i < `N; i++) begin
                            fan_register[level][i] = fan_data[level][i];
                            fan_out_idx_register[level][i] = fan_out_idx[level][i];
                        end
                        for (int i = (1 << level) - 1; i < `N; i += step) begin
                            if (add_En[level][i]) begin
                                fan_register[level][i] = 0;
                                if (bypass_En[level][left_sel[level][i]]) fan_register[level][i] += fan_data[level][left_sel[level][i]+1];
                                else fan_register[level][i] += fan_data[level][left_sel[level][i]];
                                fan_register[level][i] += fan_data[level][right_sel[level][i]];
                                fan_out_idx_register[level][vecID[level][i]] = i;
                            end
                        end
                        for (int i = 0; i < `N; i++) begin
                            zero_register[level+1][i] <= zero_register[level][i];
                            out_idx_register[level+1][i] <= out_idx_register[level][i];
                            split_register[level+1][i] <= split_register[level][i];
                            fan_data[level+1][i] <= fan_register[level][i];
                            fan_out_idx[level+1][i] <= fan_out_idx_register[level][i];
                            vecID[level+1][i] <= vecID[level][i];
                            add_En[level+1][i] <= add_En[level][i];
                            bypass_En[level+1][i] <= bypass_En[level][i];
                            adderLvl[level+1][i] <= adderLvl[level][i];
                            left_sel[level+1][i] <= left_sel[level][i];
                            right_sel[level+1][i] <= right_sel[level][i];
                        end
                        out_scale_register[level+1] <= out_scale_register[level];

                        if (level == NUM_LEVELS-1) begin
                            for (int i = 0; i < `N; i++) begin
                                out_data_register[i] = 0;
                                if (split_register[level][i]) begin
                                    out_data_register[i] = fan_register[level][fan_out_idx_register[level][vecID[level][i]]];
                                    if (vecID[level][i] == 0) out_data_register[i] += halo_in;
                                end
                            end
                            for (int i = out_scale_register[level][0]; i <= out_scale_register[level][1]; i++)
                                if (!zero_register[level][i])
                                    out_data[i] = out_data_register[out_idx_register[level][i]];
                                else out_data[i] = 0;
                            flag = 0;
                            for (int i = 0; i < `N; ++i)
                                if (split_register[level][i]) begin
                                    flag = 1;
                                    break;
                                end
                            if (flag && !split_register[level][`N-1]) begin
                                //halo_out = fan_register[level][fan_out_idx_register[level][vecID[level][`N-1]]];
                                halo_out = 0;
                            end else halo_out = 0;
                        end
                    end
                end
            end
        end
    endgenerate
    assign num_el = `N;
    assign delay = NUM_LEVELS;

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
    int valid = 0;
    //assign valid = valid || lhs_start;
    always_ff @(posedge lhs_start) valid += 1;
    

    // split 用于记录 lhs_ptr 的变化
    logic split[`N-1:0];
    logic red_split[`N-1:0];
    logic [`lgN-1:0] current_pos_ptr;
    logic [`lgN-1:0] current_row_data;
    logic [`lgN-1:0] current_row_output;
    logic [`lgN-1:0] out_idx[`N-1:0];
    logic [`lgN-1:0] out_scale[2];
    int time_counter;
    // current_row_data 记录当前data的部分和求到哪一行了
    // current_row_output 记录当前out_idx输出到哪一行了
    logic flag = 0;
    always_ff @(posedge clock or posedge reset) begin
        if (reset || lhs_start) begin
            split = '{default: 0};
            current_pos_ptr = 0;
            current_row_data = 0;
            current_row_output = 0;
            out_idx = '{default: 0};
            time_counter <= 0;
            out_scale[0] = 0;
            out_scale[1] = 0;
        end else if (valid) begin
            split = '{default: 0};
            out_scale[0] = current_row_output;
            for (int i = 0; i < `N; ++i) begin
                if (lhs_ptr[current_pos_ptr] >= current_row_data * `N &&
                lhs_ptr[current_pos_ptr] < (current_row_data + 1) * `N) begin
                    split[lhs_ptr[current_pos_ptr] % `N] = 1;
                    out_idx[current_row_output++] = (lhs_ptr[current_pos_ptr]) % `N;
                    current_pos_ptr++;
                end
            end
            out_scale[1] = current_row_output-1;

            if (current_row_output == 0) valid = 0 || lhs_start;
            current_row_data++;
        end else begin
            // split = '{default: 0};
            // current_pos_ptr = 0;
            // current_row_data = 0;
            // current_row_output = 0;
            // out_idx = '{default: 0};
            // time_counter <= 0;
            // out_scale[0] = 0;
            // out_scale[1] = 0;
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
        else halo_in <= halo_out;
    end

    logic zero[`N-1:0];
    always_comb begin
        out = '{default: 0};
        zero = '{default: 0};
        for (int i = 1; i < `N; ++i)
            if (lhs_ptr[i] == lhs_ptr[i - 1])
                zero[i] = 1;
        for (int i = 0; i < `N; i++)
            out[i] = out_buffer[i];
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
        .halo_in(halo_in),
        .zero(zero),
        .out_scale(out_scale)
    );
        
    assign delay = red_delay + 1;
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

    // Input Part
    data_t rhs_buffer [1:0][`N-1:0][`N-1:0]; // RHS buffer - stores full NxN matrix
    logic [$clog2(`N>>2):0] rhs_block_counter; // Counter for blocks of 4 rows during RHS loading progress
    logic [1:0] rhs_buffer_counter;
    logic rhs_transfer_done; // Flag to indicate when RHS loading is done and transfered to processing buffer
    
    typedef enum logic [1:0] {
        RHS_IDLE,
        RHS_LOAD,
        RHS_TRANSFER
    } rhs_state_t;
    rhs_state_t rhs_state, rhs_state_next;

    // Next-state logic
    always_comb begin
        rhs_state_next = rhs_state;
        rhs_ready      = 0;
        case (rhs_state)
            RHS_IDLE: begin
                rhs_ready      = 1;
                if (rhs_start) rhs_state_next = RHS_LOAD;
            end

            RHS_LOAD: begin
                rhs_ready = 0;
                if (rhs_block_counter == (`N>>2)) begin
                    rhs_state_next = RHS_IDLE;
                end
            end

            RHS_TRANSFER: begin
                rhs_state_next = RHS_IDLE;
            end
        endcase
    end

    // Data‐path for reading 4 rows at a time
    always_ff @(posedge clock or posedge reset) begin
        if (reset) begin
            rhs_state           <= RHS_IDLE;
            rhs_block_counter   <= 0;
            rhs_buffer_counter  <= 0;
            rhs_transfer_done   = 0;
        end else begin
            case (rhs_state)
                RHS_IDLE: begin
                    rhs_block_counter   <= 0;
                    rhs_transfer_done   = 0;
                end

                RHS_LOAD: begin
                    for (int i = 0; i < 4; i++)
                        for (int j = 0; j < `N; j++)
                            rhs_buffer[1][j][rhs_block_counter * 4 + i] <= rhs_data[i][j];
                    
                    if (rhs_block_counter < (`N>>2)) rhs_block_counter <= rhs_block_counter + 1;
                    else begin
                        rhs_buffer_counter <= rhs_buffer_counter + 1;
                        if (!rhs_buffer_counter) begin
                            for (int i = 0; i < `N; i++)
                                for (int j = 0; j < `N; j++)
                                    rhs_buffer[0][i][j] <= rhs_buffer[1][i][j];
                            for (int i = 0; i < `N; i++)
                                for (int j = 0; j < `N; j++)
                                    rhs_buffer[1][i][j] <= 0;
                        end
                    end
                end
            endcase
            
            rhs_state <= rhs_state_next;
        end
    end


    //Processing Part
    logic [$clog2(`N):0] processing_counter; // Counter for processing time, deciding when to output
    int pe_delay;
    data_t pe_outputs [`N-1:0][`N-1:0]; // PE Output buffers

    typedef enum logic [1:0] {
        LHS_IDLE,
        LHS_PROCESS
    } lhs_state_t;
    lhs_state_t lhs_state, lhs_state_next;

    assign lhs_ready_ws = lhs_ready_ns;
    assign lhs_ready_os = lhs_ready_ws;
    assign lhs_ready_wos = lhs_ready_os;
    always_comb begin
        lhs_state_next = lhs_state;
        case (lhs_state)
            LHS_IDLE: begin
                if (lhs_start) begin
                    lhs_state_next = LHS_PROCESS;
                end

                lhs_ready_ns = (rhs_buffer_counter > 0);
            end

            LHS_PROCESS: begin
                lhs_ready_ns = 0;
                if (processing_counter == (`N-1)) lhs_state_next = LHS_IDLE;
            end
        endcase
    end

    // Actual multiply logic + writing to out_buffer
    always_ff @(posedge clock or posedge reset) begin
        if (reset) begin
            lhs_state           <= LHS_IDLE;
            processing_counter  = 0;
        end else begin
            case (lhs_state)
                LHS_IDLE: begin
                    processing_counter = 0;
                    if (lhs_start && lhs_os) output_stationary++;
                end

                LHS_PROCESS: begin
                    // For each cycle, feed part of LHS and the selected RHS buffer into the PEs
                    if (processing_counter == `N-1) begin
                        if (!lhs_ws) begin
                            for (int i = 0; i < `N; i++)
                                for (int j = 0; j < `N; j++)
                                    rhs_buffer[0][i][j] <= rhs_buffer[1][i][j];
                            for (int i = 0; i < `N; i++)
                                for (int j = 0; j < `N; j++)
                                    rhs_buffer[1][i][j] <= 0;
                            rhs_buffer_counter <= rhs_buffer_counter - 1;
                        end
                    end else processing_counter++;
                end
            endcase

            lhs_state <= lhs_state_next;
        end
    end

    // PE array instantiation - N PEs, each processing one column
    genvar i;
    generate
        for (i = 0; i < `N; i++) begin : pe_array
            PE pe_inst (
                .clock(clock),
                .reset(reset),
                .lhs_start(lhs_start),
                .lhs_ptr(lhs_ptr),
                .lhs_col(lhs_col),
                .lhs_data(lhs_data),
                .rhs(rhs_buffer[0][i]),
                .out(pe_outputs[i]),
                .delay(pe_delay),
                .num_el()
            );
        end
    endgenerate


    // Output Part
    logic [$clog2(`N>>2):0] output_block_counter; // Counter for blocks of 4 rows during output progress
    logic [1:0] output_buffer_counter;
    data_t out_buffer [1:0][`N-1:0][`N-1:0]; // Output buffer
    logic [$clog2(`N):0] output_receiving_counter;
    logic [`N:0] output_stationary;
    logic out_valid;

    // Output buffer logic
    typedef enum logic [1:0] {
        OUT_IDLE,
        OUT_RECEIVING
    } out_state_t;
    out_state_t out_state, out_state_next;

    always_comb begin
        out_state_next = out_state;
        case (out_state)
            OUT_IDLE: begin
                if (processing_counter >= pe_delay-2) begin
                    out_state_next = OUT_RECEIVING;
                end
            end

            OUT_RECEIVING: begin
                if (output_receiving_counter == `N) begin
                    if (processing_counter >= pe_delay-2) begin
                        out_state_next = OUT_RECEIVING;
                    end else out_state_next = OUT_IDLE;
                end
            end
        endcase
    end


    
    generate
        for (genvar i = 0; i < `N; ++i) begin
            data_t t_out[`N-1:0];
            data_t t_out_buffer[`N-1:0];
            for (genvar j = 0; j < `N; ++j) assign t_out[j] = pe_outputs[j][i];
            for (genvar j = 0; j < `N; ++j) assign t_out_buffer[j] = out_buffer[1][i][j];
        end
    endgenerate

    // Sending out 4 rows at a time from the chosen out_buffer
    always_ff @(posedge clock or posedge reset) begin
        if (reset) begin
            out_state <= OUT_IDLE;
            output_receiving_counter <= 0;
            output_stationary = 0;
            output_buffer_counter = 0;
            out_ready <= 0;
            out_valid <= 0;
        end else begin
            case (out_state)
                OUT_IDLE: begin
                    output_receiving_counter <= 0;
                end

                OUT_RECEIVING: begin
                    output_receiving_counter <= output_receiving_counter + 1;
                    for (int row = 0; row < `N; row ++)
                        for (int col = 0; col < `N; col++)
                            out_buffer[1][row][col] <= pe_outputs[col][row];
                    if (output_receiving_counter == `N) begin
                        output_buffer_counter++;
                        // out_valid <= (!output_stationary);
                        // for (int i = 0; i < `N; i++)
                        //     for (int j = 0; j < `N; j++)
                        //         out_buffer[0][i][j] <= out_buffer[1][i][j];
                        if (!output_stationary || output_buffer_counter == 1) begin
                            out_valid <= (!output_stationary);
                            for (int i = 0; i < `N; i++)
                                for (int j = 0; j < `N; j++)
                                    out_buffer[0][i][j] <= out_buffer[1][i][j];
                        end else begin
                            for (int i = 0; i < `N; i++)
                                for (int j = 0; j < `N; j++)
                                    out_buffer[0][i][j] <= out_buffer[0][i][j] + out_buffer[1][i][j];
                            output_stationary--;
                            output_buffer_counter--;
                            out_valid <= (!output_stationary);
                        end
                        
                        output_receiving_counter <= 0;
                    end
                end
            endcase

            out_state <= out_state_next;
        end
    end

    always_ff @(posedge clock or posedge reset) begin
        if (reset) begin
            output_block_counter = 0;
        end else begin
            if (out_valid) begin
                if (!output_block_counter) out_ready <= 1;
                else out_ready <= 0;

                for (int row = 0; row < 4; row++) begin
                    for (int col = 0; col < `N; col++) begin
                        out_data[row][col] = out_buffer[0][output_block_counter*4 + row][col];
                    end
                end
                output_block_counter++;
                if (output_block_counter == (`N>>2)) begin
                    output_buffer_counter--;
                    out_valid <= 0;
                    output_block_counter = 0;
                end
            end else begin
                out_ready <= 0;
            end
        end
    end

    generate
        for(genvar i = 0; i < `N; i++) begin
            data_t out_dta[`N-1:0];
            for(genvar j = 0; j < `N; j++) begin
                assign out_dta[j] = out_buffer[0][i][j];
            end
        end

    endgenerate

endmodule