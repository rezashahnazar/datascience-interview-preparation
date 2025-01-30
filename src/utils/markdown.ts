import fs from "fs";
import path from "path";
import matter from "gray-matter";

const docsDirectory = path.join(process.cwd(), "docs");

export function getAllDocs() {
  const fileNames = fs.readdirSync(docsDirectory).filter((fileName) => {
    const fullPath = path.join(docsDirectory, fileName);
    return fs.statSync(fullPath).isFile() && fileName.endsWith(".md");
  });

  const allDocsData = fileNames.map((fileName) => {
    const id = fileName.replace(/\.md$/, "");
    const fullPath = path.join(docsDirectory, fileName);
    const fileContents = fs.readFileSync(fullPath, "utf8");
    const matterResult = matter(fileContents);

    return {
      id,
      title:
        fileName === "main.md"
          ? "Introduction"
          : `Day ${id.replace("day", "")}`,
      ...matterResult.data,
      content: matterResult.content,
    };
  });

  return allDocsData;
}

export function getDocBySlug(slug: string) {
  try {
    const fullPath = path.join(docsDirectory, `${slug}.md`);
    if (!fs.existsSync(fullPath)) {
      throw new Error(`File not found: ${slug}.md`);
    }
    const fileContents = fs.readFileSync(fullPath, "utf8");
    const { data, content } = matter(fileContents);

    return {
      slug,
      title:
        slug === "main" ? "Introduction" : `Day ${slug.replace("day", "")}`,
      content,
      ...data,
    };
  } catch (error) {
    console.error(`Error reading file ${slug}.md:`, error);
    throw error;
  }
}

export function getAllDocSlugs() {
  const fileNames = fs.readdirSync(docsDirectory).filter((fileName) => {
    const fullPath = path.join(docsDirectory, fileName);
    return fs.statSync(fullPath).isFile() && fileName.endsWith(".md");
  });

  return fileNames.map((fileName) => {
    return {
      params: {
        slug: fileName.replace(/\.md$/, ""),
      },
    };
  });
}
