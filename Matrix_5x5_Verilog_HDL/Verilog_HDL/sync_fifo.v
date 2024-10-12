`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/07/10 21:37:58
// Design Name: 
// Module Name: sync_fifo
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module sync_fifo#(parameter FIFO_SIZE=64, BUF_WIDTH=8) (
    //FIFO的数据位宽默认为8bit
   
    input                      i_clk,//输入时钟
    input                      i_rst_n,//复位信号
    input                      i_w_en,//写使能信号
    input                      i_r_en,//读使能信号
    input      [BUF_WIDTH-1:0] i_data,//写入数据

    output 	   [7:0]		   o_data,//读出数据
    output                     o_buf_empty,//FIFO空标志
    output    				   o_fifo_cnt,
    output                     o_buf_full 
);//FIFO满标志

/************************************************ reg **********************************************************/	
	reg 		[BUF_WIDTH-1:0] 		r_data										;	
    reg 		[7:0] 					r_fifo_cnt									; 
    reg [$clog2(FIFO_SIZE)-1:0] 		r_ptr,w_ptr									;  //数据指针为3位宽度，0-7索引，8个数据深度，循环指针0-7-0-7
    reg [BUF_WIDTH-1:0] 				buf_mem					[0:FIFO_SIZE-1]		; //定义FIFO大小	
/************************************************ wire **********************************************************/		
	wire								r_buf_empty	;
	wire								r_buf_full	;
    wire 		[7:0] 					o_fifo_cnt	;  
	

/************************************************ assign **********************************************************/	
assign			o_data			=		r_data		;
assign			o_fifo_cnt		=		r_fifo_cnt	;
assign			o_buf_empty		=		r_buf_empty	;
assign			o_buf_full		=		r_buf_full	;
//判断空满
assign r_buf_empty=(r_fifo_cnt==8'd0)?1'b1:1'b0;
assign r_buf_full=(r_fifo_cnt==8'd64)?1'b1:1'b0;
/************************************************ always_combinational **********************************************************/	
    //always@(posedge i_clk or negedge i_rst_n) //读数据
	always@(*)
        begin
            if(!i_rst_n)
                r_data<=8'd0;
            else if(!r_buf_empty&&i_r_en)//当没空且读使能有效
                r_data<=buf_mem[r_ptr];
			else
				r_data<=r_data;
        end

    always@(posedge i_clk)  //写数据
        begin
            if(!r_buf_full&&i_w_en)//当没满且写使能有效
			begin
                buf_mem[w_ptr]<=i_data;
				
			end
        end
/************************************************ always_temporal **********************************************************/
    always@(posedge i_clk or negedge i_rst_n) //用于修改计数器
        begin
            if(!i_rst_n)
                r_fifo_cnt<=4'd0;
            else if((!r_buf_full&&i_w_en)&&(!r_buf_empty&&i_r_en)) //同时读写，且既没空也没满，计数器不变
			//else if((i_w_en)&&(i_r_en)) //同时读写，且既没空也没满，计数器不变
                r_fifo_cnt<=r_fifo_cnt;
				//r_fifo_cnt_a<=r_fifo_cnt_a+1;
            else if(!r_buf_full&&i_w_en) //写数据，且没满，计数器加1
                r_fifo_cnt<=r_fifo_cnt+1;
				//r_fifo_cnt_a<=r_fifo_cnt_a+1;
            else if(!r_buf_empty&&i_r_en) //读数据，且没空，计数器减1
                r_fifo_cnt<=r_fifo_cnt-1;
            else
                r_fifo_cnt <= r_fifo_cnt; //其他情况，计数器不变
        end

    always@(posedge i_clk or negedge i_rst_n) //读写地址指针变化
        begin
            if(!i_rst_n) begin
                w_ptr <= 0;
                r_ptr <= 0;
            end
            else begin
                if(!r_buf_full&&i_w_en) // 写数据，地址加1，溢出后自动回到0开始
                        w_ptr <= w_ptr + 1;
                if(!r_buf_empty&&i_r_en) // 读数据，地址加1，溢出后自动回到0开始
						r_ptr <= r_ptr + 1;
            end
        end

endmodule
