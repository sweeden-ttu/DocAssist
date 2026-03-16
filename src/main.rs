use docassist::pdf_service_server::PdfService;
use docassist::{ExtractTextRequest, ExtractTextResponse, GetPageCountRequest, GetPageCountResponse, RenderPageRequest, RenderPageResponse};
use pdfium_render::prelude::*;
use tonic::transport::Server;
use tonic::{Request, Response, Status};

mod docassist {
    tonic::include_proto!("docassist");
}

pub struct PdfServiceImpl {
    pdfium: Pdfium,
}

impl PdfServiceImpl {
    pub fn new() -> Result<Self, String> {
        let pdfium = Pdfium::default();
        Ok(Self { pdfium })
    }
}

#[tonic::async_trait]
impl PdfService for PdfServiceImpl {
    async fn render_page(
        &self,
        request: Request<RenderPageRequest>,
    ) -> Result<Response<RenderPageResponse>, Status> {
        let req = request.into_inner();
        
        let document = self.pdfium.load_pdf_from_byte_vec(req.pdf_data, None)
            .map_err(|e| Status::internal(e.to_string()))?;
        
        let page_index = req.page_index as usize;
        let pages = document.pages();
        let page = pages.iter()
            .nth(page_index)
            .ok_or_else(|| Status::not_found("Page not found"))?;
        
        let width = if req.width > 0 { req.width as i32 } else { 612 };
        let height = if req.height > 0 { req.height as i32 } else { 792 };
        
        let render_config = PdfRenderConfig::new()
            .set_target_width(width)
            .set_maximum_height(height);
        
        let image = page.render_with_config(&render_config)
            .map_err(|e| Status::internal(e.to_string()))?
            .as_image()
            .map_err(|e| Status::internal(e.to_string()))?
            .into_rgb8();
        
        let mut png_data = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut png_data);
        image.write_to(&mut cursor, image::ImageFormat::Png)
            .map_err(|e| Status::internal(e.to_string()))?;
        
        Ok(Response::new(RenderPageResponse {
            image_data: png_data,
            actual_width: width as u32,
            actual_height: height as u32,
        }))
    }

    async fn get_page_count(
        &self,
        request: Request<GetPageCountRequest>,
    ) -> Result<Response<GetPageCountResponse>, Status> {
        let req = request.into_inner();
        
        let document = self.pdfium.load_pdf_from_byte_vec(req.pdf_data, None)
            .map_err(|e| Status::internal(e.to_string()))?;
        
        let page_count = document.pages().len() as u32;
        
        Ok(Response::new(GetPageCountResponse { page_count }))
    }

    async fn extract_text(
        &self,
        request: Request<ExtractTextRequest>,
    ) -> Result<Response<ExtractTextResponse>, Status> {
        let req = request.into_inner();
        
        let document = self.pdfium.load_pdf_from_byte_vec(req.pdf_data, None)
            .map_err(|e| Status::internal(e.to_string()))?;
        
        let page_index = req.page_index as usize;
        let pages = document.pages();
        let page = pages.iter()
            .nth(page_index)
            .ok_or_else(|| Status::not_found("Page not found"))?;
        
        let text = page.text()
            .map_err(|e| Status::internal(e.to_string()))?
            .to_string();
        
        Ok(Response::new(ExtractTextResponse { text }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    let pdf_service = PdfServiceImpl::new()?;
    
    let addr = "[::1]:50051".parse()?;
    
    Server::builder()
        .add_service(docassist::pdf_service_server::PdfServiceServer::new(pdf_service))
        .serve(addr)
        .await?;
    
    Ok(())
}
