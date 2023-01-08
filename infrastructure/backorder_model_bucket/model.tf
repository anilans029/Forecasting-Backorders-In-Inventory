
resource "aws_s3_bucket" "model_bucket" {
  bucket        = "${var.model_bucket_name}"
  force_destroy = var.force_destroy_bucket
}