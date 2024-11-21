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
    //FIFO������λ��Ĭ��Ϊ8bit
   
    input                      i_clk,//����ʱ��
    input                      i_rst_n,//��λ�ź�
    input                      i_w_en,//дʹ���ź�
    input                      i_r_en,//��ʹ���ź�
    input      [BUF_WIDTH-1:0] i_data,//д������

    output 	   [7:0]		   o_data,//��������
    output                     o_buf_empty,//FIFO�ձ�־
    output    				   o_fifo_cnt,
    output                     o_buf_full 
);//FIFO����־

/************************************************ reg **********************************************************/	
	reg 		[BUF_WIDTH-1:0] 		r_data										;	
    reg 		[7:0] 					r_fifo_cnt									; 
    reg [$clog2(FIFO_SIZE)-1:0] 		r_ptr,w_ptr									;  //����ָ��Ϊ3λ��ȣ�0-7������8��������ȣ�ѭ��ָ��0-7-0-7
    reg [BUF_WIDTH-1:0] 				buf_mem					[0:FIFO_SIZE-1]		; //����FIFO��С	
/************************************************ wire **********************************************************/		
	wire								r_buf_empty	;
	wire								r_buf_full	;
    wire 		[7:0] 					o_fifo_cnt	;  
	

/************************************************ assign **********************************************************/	
assign			o_data			=		r_data		;
assign			o_fifo_cnt		=		r_fifo_cnt	;
assign			o_buf_empty		=		r_buf_empty	;
assign			o_buf_full		=		r_buf_full	;
//�жϿ���
assign r_buf_empty=(r_fifo_cnt==8'd0)?1'b1:1'b0;
assign r_buf_full=(r_fifo_cnt==8'd64)?1'b1:1'b0;
/************************************************ always_combinational **********************************************************/	
    //always@(posedge i_clk or negedge i_rst_n) //������
	always@(*)
        begin
            if(!i_rst_n)
                r_data<=8'd0;
            else if(!r_buf_empty&&i_r_en)//��û���Ҷ�ʹ����Ч
                r_data<=buf_mem[r_ptr];
			else
				r_data<=r_data;
        end

    always@(posedge i_clk)  //д����
        begin
            if(!r_buf_full&&i_w_en)//��û����дʹ����Ч
			begin
                buf_mem[w_ptr]<=i_data;
				
			end
        end
/************************************************ always_temporal **********************************************************/
    always@(posedge i_clk or negedge i_rst_n) //�����޸ļ�����
        begin
            if(!i_rst_n)
                r_fifo_cnt<=4'd0;
            else if((!r_buf_full&&i_w_en)&&(!r_buf_empty&&i_r_en)) //ͬʱ��д���Ҽ�û��Ҳû��������������
			//else if((i_w_en)&&(i_r_en)) //ͬʱ��д���Ҽ�û��Ҳû��������������
                r_fifo_cnt<=r_fifo_cnt;
				//r_fifo_cnt_a<=r_fifo_cnt_a+1;
            else if(!r_buf_full&&i_w_en) //д���ݣ���û������������1
                r_fifo_cnt<=r_fifo_cnt+1;
				//r_fifo_cnt_a<=r_fifo_cnt_a+1;
            else if(!r_buf_empty&&i_r_en) //�����ݣ���û�գ���������1
                r_fifo_cnt<=r_fifo_cnt-1;
            else
                r_fifo_cnt <= r_fifo_cnt; //�������������������
        end

    always@(posedge i_clk or negedge i_rst_n) //��д��ַָ��仯
        begin
            if(!i_rst_n) begin
                w_ptr <= 0;
                r_ptr <= 0;
            end
            else begin
                if(!r_buf_full&&i_w_en) // д���ݣ���ַ��1��������Զ��ص�0��ʼ
                        w_ptr <= w_ptr + 1;
                if(!r_buf_empty&&i_r_en) // �����ݣ���ַ��1��������Զ��ص�0��ʼ
						r_ptr <= r_ptr + 1;
            end
        end

endmodule
