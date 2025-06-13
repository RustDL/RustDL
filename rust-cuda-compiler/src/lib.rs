extern crate proc_macro;
use proc_macro::TokenStream;

use quote::quote;
use syn::Stmt;
use syn::visit::Visit;

struct IRGenerator {
    pub ir: String,
    pub temp_idx: usize,
}

impl IRGenerator {
    fn new() -> Self {
        IRGenerator { ir: String::new(), temp_idx: 0 }
    }
    fn next_temp(&mut self) -> String {
        let name = format!("%{}", self.temp_idx);
        self.temp_idx += 1;
        name
    }
}

impl<'ast> Visit<'ast> for IRGenerator {
    fn visit_block(&mut self, node: &'ast syn::Block) {
        for stmt in &node.stmts {
            self.visit_stmt(stmt);
        }
    }

    fn visit_expr_assign(&mut self, node: &'ast syn::ExprAssign) {
        let left = match &*node.left {
            syn::Expr::Path(p) => format!("%{}", quote::quote!(#p)),
            _ => "<unsupported>".to_string(),
        };

        if let syn::Expr::Binary(expr_bin) = &*node.right {
            self.visit_expr_binary(expr_bin);
            let temp = format!("%{}", self.temp_idx - 1);
            self.ir.push_str(&format!("  {} = {}\n", left, temp));
        } else if let syn::Expr::Lit(lit) = &*node.right {
            self.ir.push_str(&format!("  {} = {}\n", left, quote::quote!(#lit)));
        } else if let syn::Expr::Path(p) = &*node.right {
            self.ir.push_str(&format!("  {} = %{}\n", left, quote::quote!(#p)));
        }
    }

    fn visit_expr_binary(&mut self, node: &'ast syn::ExprBinary) {
        let left = match &*node.left {
            syn::Expr::Path(p) => format!("%{}", quote::quote!(#p).to_string()),
            syn::Expr::Lit(l) => quote::quote!(#l).to_string(),
            syn::Expr::Binary(b) => {
                self.visit_expr_binary(b);
                format!("%{}", self.temp_idx - 1)
            },
            syn::Expr::Paren(body) => {
                self.visit_expr(&*body.expr);
                format!("%{}", self.temp_idx - 1)
            },
            _ => "<unsupported>".to_string(),
        };

        let right = match &*node.right {
            syn::Expr::Path(p) => format!("%{}", quote::quote!(#p).to_string()),
            syn::Expr::Lit(l) => quote::quote!(#l).to_string(),
            syn::Expr::Binary(b) => {
                self.visit_expr_binary(b);
                format!("%{}", self.temp_idx - 1)
            },
            syn::Expr::Paren(body) => {
                self.visit_expr(&*body.expr);
                format!("%{}", self.temp_idx - 1)
            },
            _ => "<unsupported>".to_string(),
        };

        let op = match &node.op {
            syn::BinOp::Add(_) => "add nsw i32",
            syn::BinOp::Sub(_) => "sub nsw i32",
            syn::BinOp::Mul(_) => "mul nsw i32",
            syn::BinOp::Div(_) => "sdiv i32",
            syn::BinOp::Eq(_) => "icmp eq i32",
            syn::BinOp::Ne(_) => "icmp ne i32",
            syn::BinOp::Lt(_) => "icmp slt i32",
            syn::BinOp::Le(_) => "icmp sle i32",
            syn::BinOp::Gt(_) => "icmp sgt i32",
            syn::BinOp::Ge(_) => "icmp sge i32",
            _ => "unknown i32",
        };

        let temp = self.next_temp();
        self.ir.push_str(&format!("  {} = {} {}, {}\n", temp, op, left, right));
    }
    fn visit_expr_if(&mut self, node: &'ast syn::ExprIf) {
        let cond = match &*node.cond {
            syn::Expr::Path(p) => format!("%{}", quote::quote!(#p)),
            syn::Expr::Lit(l) => quote::quote!(#l).to_string(),
            _ => "<unsupported>".to_string(),
        };
        let then_label = self.next_temp();
        let else_label = self.next_temp();
        let end_label = self.next_temp();

        self.ir.push_str(&format!("  br i1 {}, label {}, label {}\n", cond, then_label, else_label));
        self.ir.push_str(&format!("{}:\n", then_label));
        for stmt in &node.then_branch.stmts {
            self.visit_stmt(stmt);
        }
        self.ir.push_str(&format!("  br label {}\n", end_label));
        self.ir.push_str(&format!("{}:\n", else_label));
        if let Some((_, else_branch)) = &node.else_branch {
            match &**else_branch {
                syn::Expr::Block(b) => {
                    for stmt in &b.block.stmts {
                        self.visit_stmt(stmt);
                    }
                }
                _ => {}
            }
        }
        self.ir.push_str(&format!("  br label {}\n", end_label));
        self.ir.push_str(&format!("{}:\n", end_label));
    }
    fn visit_expr_return(&mut self, node: &'ast syn::ExprReturn) {
        if let Some(expr) = &node.expr {
            match &**expr {
                syn::Expr::Path(p) => {
                    self.ir.push_str(&format!("  ret i32 %{}\n", quote::quote!(#p)));
                }
                syn::Expr::Lit(l) => {
                    self.ir.push_str(&format!("  ret i32 {}\n", quote::quote!(#l)));
                }
                syn::Expr::Binary(b) => {
                    self.visit_expr_binary(b);
                    let temp = format!("%{}", self.temp_idx - 1);
                    self.ir.push_str(&format!("  ret i32 {}\n", temp));
                }
                _ => self.ir.push_str("  ret void\n"),
            }
        } else {
            self.ir.push_str("  ret void\n");
        }
    }
    fn visit_local(&mut self, node: &'ast syn::Local) {
        if let Some((_, expr)) = &node.init {
            match &node.pat {
                syn::Pat::Ident(ident) => {
                    let var = ident.ident.to_string();
                    if let syn::Expr::Binary(expr_bin) = &**expr {
                        self.visit_expr_binary(expr_bin);
                        let temp = format!("%{}", self.temp_idx - 1);
                        self.ir.push_str(&format!("  %{} = {}\n", var, temp));
                    } else if let syn::Expr::Lit(lit) = &**expr {
                        self.ir.push_str(&format!("  %{} = {}\n", var, quote::quote!(#lit)));
                    } else if let syn::Expr::Path(p) = &**expr {
                        self.ir.push_str(&format!("  %{} = %{}\n", var, quote::quote!(#p)));
                    }
                }
                _ => {}
            }
        }
    }
    fn visit_stmt(&mut self, node: &'ast syn::Stmt) {
        match &*node {
            Stmt::Local(local) => {
                self.visit_local(local)
            }            
            Stmt::Expr(expr) => {
                self.visit_expr(expr);
            }
            Stmt::Semi(expr, _) => {
                self.visit_expr(expr);
            }
            Stmt::Item(_) => {}
        }
    }
}

#[proc_macro_attribute]
pub fn cuda(_args: TokenStream, input: TokenStream) -> TokenStream {
    let mut item_fn = syn::parse_macro_input!(input as syn::ItemFn);
    let fn_name = item_fn.sig.ident.to_string();
    let params: Vec<(String, String)> = item_fn
        .sig
        .inputs
        .iter()
        .filter_map(|arg| {
            if let syn::FnArg::Typed(pat_type) = arg {
                let pat = &pat_type.pat;
                let ty = &pat_type.ty;
                let name = quote::quote!(#pat).to_string();
                let ty_str = if quote::quote!(#ty).to_string().contains("i32") { "i32" } else { "unknown" };
                Some((name, ty_str.to_string()))
            } else {
                None
            }
        })
        .collect();
    let ret_ty = match &item_fn.sig.output {
        syn::ReturnType::Type(_, ty) => {
            if quote::quote!(#ty).to_string().contains("i32") {
                "i32"
            } else {
                "void"
            }
        }
        _ => "void",
    };
    // 访问函数体，生成IR
    let mut visitor = IRGenerator::new();
    for stmt in &item_fn.block.stmts {
        visitor.visit_stmt(stmt);
    }
    let param_str = params.iter().map(|(n, t)| format!("{} %{}", t, n)).collect::<Vec<_>>().join(", ");
    let nvvm_ir = format!("; ModuleID = 'rust_cuda_macro'\nsource_filename = \"rust_cuda_macro\"\n\ndefine {} @{}({}) {{\n{}}}\n", ret_ty, fn_name, param_str, visitor.ir);
    // 在函数体开头插入打印 NVVM IR 的语句
    let stmts = &mut item_fn.block.stmts;
    stmts.insert(
        0,
        syn::parse_quote! {
            println!("[NVVM IR for {}]:\n{}", #fn_name, #nvvm_ir);
        },
    );
    let gen = quote! {
        #item_fn
    };
    gen.into()
}
