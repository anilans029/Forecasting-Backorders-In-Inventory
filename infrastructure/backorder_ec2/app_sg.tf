resource "aws_security_group" "security_group" {
  name = var.app_sg_group_name

  ingress {
    from_port   = var.app_ingress_from_port[0]
    to_port     = var.app_ingress_to_port[0]
    protocol    = var.app_protocol
    cidr_blocks = var.app_cidr_block
  }

  ingress {
    from_port   = var.app_ingress_from_port[1]
    to_port     = var.app_ingress_to_port[1]
    protocol    = var.app_protocol
    cidr_blocks = var.app_cidr_block
  }

  ingress {
    from_port   = var.app_ingress_from_port[2]
    to_port     = var.app_ingress_to_port[2]
    protocol    = var.app_protocol
    cidr_blocks = var.app_cidr_block
  }

  ingress {
    from_port   = var.app_ingress_from_port[3]
    to_port     = var.app_ingress_to_port[3]
    protocol    = var.app_protocol
    cidr_blocks = var.app_cidr_block
  }

  ingress {
    from_port   = var.app_ingress_from_port[4]
    to_port     = var.app_ingress_to_port[4]
    protocol    = var.app_protocol
    cidr_blocks = var.app_cidr_block
  }

  ingress {
    from_port   = var.app_ingress_from_port[5]
    to_port     = var.app_ingress_to_port[5]
    protocol    = var.app_protocol
    cidr_blocks = var.app_cidr_block
  }

  ingress {
    from_port   = var.app_ingress_from_port[6]
    to_port     = var.app_ingress_to_port[6]
    protocol    = var.app_protocol
    cidr_blocks = var.app_cidr_block
  }

  ingress {
    from_port   = var.app_ingress_from_port[7]
    to_port     = var.app_ingress_to_port[7]
    protocol    = var.app_protocol
    cidr_blocks = var.app_cidr_block
  }

  ingress {
    from_port   = var.app_ingress_from_port[8]
    to_port     = var.app_ingress_to_port[8]
    protocol    = var.app_protocol
    cidr_blocks = var.app_cidr_block
  }

  ingress {
    from_port   = var.app_ingress_from_port[9]
    to_port     = var.app_ingress_to_port[9]
    protocol    = var.app_protocol
    cidr_blocks = var.app_cidr_block
  }

  egress {
    from_port   = var.app_egress_from_port
    to_port     = var.app_egress_to_port
    protocol    = var.app_protocol
    cidr_blocks = var.app_cidr_block
  }
  

  tags = {
    Name = var.app_sg_group_name
  }
}