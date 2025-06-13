extern crate proc_macro;
use proc_macro::TokenStream;

use quote::quote;
use quote::ToTokens;

#[proc_macro_attribute]
pub fn cuda(_args: TokenStream, input: TokenStream) -> TokenStream {
    let mut item_fn = syn::parse_macro_input!(input as syn::ItemFn);
    let fn_code = item_fn.to_token_stream().to_string();
    let stmts = &mut item_fn.block.stmts;
    stmts.insert(0, syn::parse_quote! {
        println!("[cuda macro AST]: {}", #fn_code);
    });
    quote! {
        #item_fn
    }.into()
}