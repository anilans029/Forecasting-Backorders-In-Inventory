variable "aws_region"{
    type = string
    default = "ap-south-1"
}

variable "artifacts_bucket_name" {
  type    = string
  default = "backorder-artifact"
}

variable "aws_account_id" {
  type    = string
  default = "919666062805"
}

variable "force_destroy_bucket" {
  type    = bool
  default = true
}
