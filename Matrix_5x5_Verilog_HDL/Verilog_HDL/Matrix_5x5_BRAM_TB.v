`timescale 1ns / 1ps
module Matrix_5x5_BRAM_TB();
reg clk;
reg rst_n;
reg data_en;
initial begin
clk = 0;
rst_n = 0;
#200; rst_n = 1;
#278 data_en=1;
end

always#50 clk = ~clk;
reg ena;
reg wea;
reg [10:0] addra;
wire [10:0] addrb;
reg enb;
//reg [10:0] addrb;
wire [7:0]	c0_0;
wire [7:0]	c0_1;
wire [7:0]	c0_2;
wire [7:0]	c0_3;
wire [7:0]	c0_4;
wire [7:0]	c1_0;
wire [7:0]	c1_1;
wire [7:0]	c1_2;
wire [7:0]	c1_3;
wire [7:0]	c1_4;
wire [7:0]	c2_0;
wire [7:0]	c2_1;
wire [7:0]	c2_2;
wire [7:0]	c2_3;
wire [7:0]	c2_4;
wire [7:0]	c3_0;
wire [7:0]	c3_1;
wire [7:0]	c3_2;
wire [7:0]	c3_3;
wire [7:0]	c3_4;
wire [7:0]	c4_0;
wire [7:0]	c4_1;
wire [7:0]	c4_2;
wire [7:0]	c4_3;
wire [7:0]	c4_4;
wire  Mat3x3_data_en;
//port A:clka,ena,wea,addra,dina
reg write_state;
always @(posedge clk or negedge rst_n) begin
    if(!rst_n)
        begin
            ena <= 0;
            wea <= 0;
            addra <= 0;
            write_state <= 0;
        end
    else
    	begin
    		case(write_state)
    			1'b0:begin ena <= 1;wea <= 1;write_state <= 1'b1; end
    			1'b1:begin ena <= 1;wea <= 1;addra <= addra + 1'd1; end
    			default:write_state <= 1'b1;
    		endcase
    	end
end
/*
//read data port B:clkb,enb,addrb
reg read_state;
always @(posedge clk or negedge rst_n) begin
    if(!rst_n)
        begin
            enb <= 0;
            addrb <= 0;
            read_state <= 0;
        end
    else
    	begin
    		case(read_state)
    			1'b0:begin enb <= 1;read_state <= 1'b1; end
    			1'b1:begin enb <= 1;addrb <= addrb + 1'd1; end
    			default:read_state <= 1'b1;
    		endcase
    	end
end*/
wire [7:0]dina = addra[7:0];
wire [7:0]doutb;
wire read_en;
blk_mem_gen_0 u_BRAM (
  .clka(clk),    // input wire clka
  .ena(ena),      // input wire ena
  .wea(wea),      // input wire [0 : 0] wea
  .addra(addra),  // input wire [10 : 0] addra
  .dina(dina),    // input wire [7 : 0] dina
  .clkb(clk),    // input wire clkb
  .enb(read_en),      // input wire enb
  .addrb(addrb),  // input wire [10 : 0] addrb
  .doutb(doutb)  // output wire [7 : 0] doutb
);
/*
always@(posedge clk)
begin
	fifo_en<=read_en;
end
assign fifo_en_w=fifo_en;*/
 Matrix_5x5 u_Matrix_5_5(
  						.i_clk(clk)				,
  						.i_rst_n(rst_n)			,
						.i_stride('b0)			,
						.i_padding(0)			,
						.i_hs_num('d7)			,
						.i_vs_num('d7)			,
          				.i_data_en(data_en)		,
						.i_data_in(doutb)			,
						.o_data_addr(addrb)		,
						.o_read_en(read_en)		,
						.o_Mat5x5_data_en(Mat5x5_data_en)	,
						.o_c0_0(c0_0)             ,
						.o_c0_1(c0_1)				,
						.o_c0_2(c0_2)				, 
						.o_c0_3(c0_3)				,
						.o_c0_4(c0_4)				, 
						.o_c1_0(c1_0)				,
						.o_c1_1(c1_1)				,
						.o_c1_2(c1_2)				, 
						.o_c1_3(c1_3)				,
						.o_c1_4(c1_4)				, 						
						.o_c2_0(c2_0)				,
						.o_c2_1(c2_1)				,
						.o_c2_2(c2_2)				,
						.o_c2_3(c2_3)				,
						.o_c2_4(c2_4)				, 
						.o_c3_0(c3_0)				,
						.o_c3_1(c3_1)				,
						.o_c3_2(c3_2)				,
						.o_c3_3(c3_3)				,
						.o_c3_4(c3_4)				, 
						.o_c4_0(c4_0)				,
						.o_c4_1(c4_1)				,
						.o_c4_2(c4_2)				,
						.o_c4_3(c4_3)				,
						.o_c4_4(c4_4)				 
						);

endmodule
