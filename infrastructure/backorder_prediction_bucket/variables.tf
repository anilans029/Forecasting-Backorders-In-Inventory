variable "aws_region"{
    type = string
    default = "ap-south-1"
}

variable "prediction_bucket_name" {
  type    = string
  default = "backorder-prediction-bucket"
}

variable "aws_account_id" {
  type    = string
  default = "919666062805"
}

variable "force_destroy_bucket" {
  type    = bool
  default = true
}
