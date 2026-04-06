fn main() {
    #[cfg(target_os = "windows")]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("img/noise-gator.ico");
        res.compile().expect("Failed to compile Windows resources");
    }
}
